'''
/home/qixing.zhou/OpenHands/evaluation/benchmarks/programmer_bench_router.sh \
    --benchmark-type blank \
    --bench-path /data2/pb_shared/pb_cases_exp \
    --llm-config llm.eval_claude-sonnet-4-20250514 \
    --agent-cls CodeActAgent \
    --max-iterations 100 \
    --eval-n-limit -1 \
    --eval-num-workers 1 \
    --openhands-root /home/qixing.zhou/OpenHands \
    --dataset potatoQi-hf/ProgrammerBench \
    --data-source local \
    --split tmp \
    --n-runs 1 \
    --gpu xxx \
    --only-ids 028_test_backends_chunks_level1

消融参数:
--pre-installed
--with-browsing     开了记得 config.toml 加翻墙变量, 弄完之后 turn_off && 把环境变量注释掉
--white
--black
--without-interface
'''
import asyncio
import argparse
import os
import pandas as pd
import copy, shutil, zipfile, tarfile
from datasets import load_dataset
import json
import yaml, re, sys
import tempfile
import time
from evaluation.benchmarks.programmer_bench.utils import upsert_instance_status_jsonl, generate_evaluation_summary_report, run_pytest_and_evaluate

# 从评估工具的共享模块中导入必要的类和函数
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,       # 评估元数据的数据类
    EvalOutput,         # 评估输出的数据类
    make_metadata,      # 创建元数据对象的函数
    prepare_dataset,    # 准备数据集的函数
    run_evaluation,     # 运行评估的主函数
    reset_logger_for_multiprocessing,   # 在多进程环境中重置日志记录器的函数
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
    is_fatal_evaluation_error,
    get_metrics,
    codeact_user_response,
    calltest_aware_user_response,   # 内含原 codeact_user_response 逻辑
    set_calltest_context,  # 设置 CallTest 上下文的函数
)
from openhands.utils.async_utils import call_async_from_sync

from openhands.core.config import (
    get_llm_config_arg,     # 获取 LLM 配置参数的函数
    get_parser,
    OpenHandsConfig,
    AgentConfig,
)
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller

from openhands.controller.state.state import State

from openhands.events.action import CmdRunAction, MessageAction, Action
from openhands.events import Event, EventSource, EventStreamSubscriber
from openhands.events.serialization.event import event_to_dict

from openhands.runtime.base import Runtime

# 评测 codebase 时间
TEST_TIME = 60 * 60  # 60 min

# data: 完整 log
# core: 去掉了 "tool_call_metadata"
LOG_LEVEL = 'core'

# 工作目录, 建议设置为 workspace, 当然还是要跟 prompt 对齐
WORK_DIR = 'workspace'

# docker hub 设置
REGISTRY_URL = "crpi-hxwp25v80sd8albz.cn-shanghai.personal.cr.aliyuncs.com"
DEFAULT_DOCKER_IMAGE_PREFIX = REGISTRY_URL + '/error666/'

# 数据集设置
DATASET_NAME = "potatoQi-hf/ProgrammerBench"

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'

# RUN_WITH_BROWSING 的默认值由环境变量决定，但可被命令行参数覆盖
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': calltest_aware_user_response,
}

def get_prompt_modifications(args: argparse.Namespace) -> list[str]:
    """
    根据消融变量配置生成需要添加到 prompt.md 的修改内容。
    每个消融变量可以添加一个或多个 markdown 列表项。
    """
    modifications = []

    if args and hasattr(args, 'pre_installed') and not args.pre_installed:
        modifications.extend([
            "- You are now in a basic container environment. This environment contains only the most basic libraries (e.g. torch). In order to complete this task you need to install the other necessary environments yourself. You can find all the required dependencies in the codebase under the `/testbed` directory we gave you.\n",
            "- It is CRITICAL that you need to install the dependencies for the **test** or **dev** version of codebase, because we will test your results based on the test files in this codebase after you finish the task. If you don't install the test dependencies in this codebase, or if you miss some of them, our tests won't run and you'll get **ZERO POINTS**!\n"
        ])

    if args and hasattr(args, 'white') and args.white:
        # 动态插入测试文件名（需在 initialize_runtime 里传递 test_file_name 到 args）
        test_file_name = getattr(args, 'whitebox_test_file', None)
        if test_file_name:
            modifications.append(f"- You are given access to the test file `{test_file_name}` for this task (white-box testing). Use this information to help you solve the problem.\n")
        else:
            raise ValueError("White-box testing enabled but test file name not provided.")

    if args and hasattr(args, 'with_browsing') and args.with_browsing:
        modifications.append("- You have access to the internet and can use web browsing to help you complete the task. Make sure to look for relevant information online if needed.\n")

    return modifications

def get_instance_docker_image(
    category: str,
) -> str:
    # OpenHands version of the image
    docker_image_prefix = DEFAULT_DOCKER_IMAGE_PREFIX
    image_name = 'programmerbench.eval.x86_64.' + category
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (docker_image_prefix.rstrip('/') + '/' + image_name).lower()

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
    pre_installed: bool = False,
    cuda_visible_devices: str = None,
    enable_black_box: bool = False,  # 新增参数，控制是否启用黑盒模式
) -> OpenHandsConfig:
    # FIXME: 这里云端的名字最终再定, 目前全部用本地镜像去跑
    # 拿到镜像名字, 例如: docker.io/potaoqi/programmerbench.eval.x86_64.00 (00 号基镜像)
    # crpi-hxwp25v80sd8albz.cn-shanghai.personal.cr.aliyuncs.com/error666/programmerbench.eval.x86_64.00
    # base_container_image = get_instance_docker_image(instance['category'])

    # 根据 pre_installed 参数选择使用哪个镜像
    if pre_installed:
        # 使用预安装环境镜像
        base_container_image = instance.get('instance_image', None)
        if base_container_image:
            logger.info(f"Using pre-installed instance image: {base_container_image}")
        else:
            raise ValueError(f"instance_image not found for pre_installed instance {instance['instance_id']}")
    else:
        # 使用基础镜像
        base_container_image = instance.get('base_image', None)
        if base_container_image:
            logger.info(f"Using base image: {base_container_image}")
        else:
            raise ValueError(f"base_image not found for instance {instance['instance_id']}")

    if base_container_image is None:
        raise ValueError(f'Instance {instance["instance_id"]} does not have any image field.')
    logger.info(
        f"Using container image:\n{base_container_image}\nPlease make sure this image exists."
    )

    sandbox_config = get_default_sandbox_config_for_eval(cuda_visible_devices)
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False

    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    resource_factor = instance.resource_factor
    sandbox_config.remote_runtime_resource_factor = resource_factor

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        enable_browser=RUN_WITH_BROWSING,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    # get 'draft_editor' config if exists
    config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=ENABLE_LLM_EDITOR,
        enable_mcp=False,
        enable_call_test=enable_black_box,  # 直接使用 enable_black_box 参数
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
    )

    config.set_agent_config(agent_config)
    return config

def preprocess_lv3_repository(runtime: Runtime, test_host_path: str, repo_name: str, logger):
    """
    LV3任务的仓库预处理逻辑

    Args:
        runtime: Runtime实例，用于从容器复制文件
        test_host_path: 宿主机test目录路径
        repo_name: 仓库名称
        logger: 日志记录器
    """
    logger.info("[LV3-PREPROCESS] Starting repository preprocessing")

    # 首先从 test/path2test.txt 读取真正的 fp2 文件路径
    path2test_file = os.path.join(test_host_path, 'path2test.txt')
    if not os.path.exists(path2test_file):
        raise FileNotFoundError(f"[LV3-PREPROCESS] path2test.txt not found at {path2test_file}")

    with open(path2test_file, 'r', encoding='utf-8') as f:
        test_file_lines = [line.strip() for line in f if line.strip()]
    if not test_file_lines:
        raise ValueError(f"[LV3-PREPROCESS] path2test.txt is empty at {path2test_file}")

    # 取第一行作为真正的 fp2 文件相对路径
    fp2_relative_path = test_file_lines[0]
    fp2_filename = os.path.basename(fp2_relative_path)
    logger.info(f"[LV3-PREPROCESS] Found fp2 file path from path2test.txt: {fp2_relative_path}")

    # 0. 检查本地是否有 test/fp2_filename，若有，跳转到 step 2，若没有，进 step 1
    fp2_file_path = os.path.join(test_host_path, fp2_filename)

    if not os.path.exists(fp2_file_path):
        logger.info("[LV3-PREPROCESS] fp2 file not found, extracting from repo_name")

        # 1. 把本地 test/repo_name 下的 fp2 file 找出来，copy 到 test/ 下
        repo_path = os.path.join(test_host_path, repo_name)
        if os.path.exists(repo_path):
            # fp2_relative_path 形如 "trl/tests/test_best_of_n_sampler.py"
            # 我们需要去掉开头的 repo_name 部分，因为我们已经在 test/repo_name 目录下了
            relative_path_in_repo = fp2_relative_path[len(repo_name) + 1:]
            repo_fp2_path = os.path.join(repo_path, relative_path_in_repo)

            if os.path.exists(repo_fp2_path):
                shutil.copy2(repo_fp2_path, fp2_file_path)
                logger.info(f"[LV3-PREPROCESS] Copied fp2 file from {repo_fp2_path} to {fp2_file_path}")
            else:
                logger.warning(f"[LV3-PREPROCESS] fp2 file not found in {repo_fp2_path}")
        else:
            logger.warning(f"[LV3-PREPROCESS] Repository directory not found: {repo_path}")
    else:
        logger.info("[LV3-PREPROCESS] fp2 file already exists, skipping step 1")

    # 2. 把本地 test/repo_name 删除
    repo_path = os.path.join(test_host_path, repo_name)
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
        logger.info(f"[LV3-PREPROCESS] Removed existing repository directory: {repo_path}")
    time.sleep(1)  # 确保文件系统稳定

    # 3. 把容器里 /root/my_repo1 copy 到本地 test/ 下，改名 /test/repo_name, 并删掉里面的 .git 文件夹
    try:
        # 从容器中拉取 /root/my_repo1 目录
        my_repo1_archive_path = runtime.copy_from('/root/my_repo1')
        logger.info(f"[LV3-PREPROCESS] Pulled /root/my_repo1 archive from container: {my_repo1_archive_path}")

        # 创建临时目录来解压文件
        temp_extract_dir = tempfile.mkdtemp(prefix='lv3_preprocess_')
        try:
            # 解压 my_repo1 内容到临时目录
            extracted = False
            try:
                if zipfile.is_zipfile(my_repo1_archive_path):
                    with zipfile.ZipFile(str(my_repo1_archive_path), 'r') as zf:
                        zf.extractall(path=temp_extract_dir)
                    logger.info("[LV3-PREPROCESS] my_repo1 archive extracted as ZIP successfully.")
                    extracted = True
                else:
                    logger.warning("[LV3-PREPROCESS] my_repo1 archive is not a valid ZIP file")
            except Exception as zip_e:
                logger.warning(f"[LV3-PREPROCESS] ZIP extraction failed for my_repo1: {zip_e}")

            # 如果ZIP失败，尝试tarfile格式
            if not extracted:
                try:
                    if tarfile.is_tarfile(my_repo1_archive_path):
                        with tarfile.open(my_repo1_archive_path, 'r:*') as tf:
                            tf.extractall(path=temp_extract_dir)
                        logger.info("[LV3-PREPROCESS] my_repo1 archive extracted as TAR successfully.")
                        extracted = True
                except Exception as tar_e:
                    logger.warning(f"[LV3-PREPROCESS] TAR extraction failed for my_repo1: {tar_e}")

            if not extracted:
                raise RuntimeError("[LV3-PREPROCESS] Failed to extract my_repo1 archive")

            # 将解压的内容移动到正确的位置（重命名为repo_name）
            # 查找解压后的目录结构
            extracted_items = os.listdir(temp_extract_dir)
            # 如果有多个文件/目录，创建repo_name目录并移动所有内容
            target_repo_path = os.path.join(test_host_path, repo_name)
            os.makedirs(target_repo_path, exist_ok=True)
            for item in extracted_items:
                src_path = os.path.join(temp_extract_dir, item)
                dst_path = os.path.join(target_repo_path, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
            logger.info(f"[LV3-PREPROCESS] Copied all extracted contents to {target_repo_path}")

            # 删除里面的 .git 文件夹
            git_dir = os.path.join(target_repo_path, '.git')
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir)
                logger.info(f"[LV3-PREPROCESS] Removed .git directory from {target_repo_path}")
            else:
                logger.info(f"[LV3-PREPROCESS] No .git directory found in {target_repo_path}")

        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_extract_dir)
            except Exception as cleanup_e:
                logger.warning(f"[LV3-PREPROCESS] Failed to cleanup temp directory {temp_extract_dir}: {cleanup_e}")

    except Exception as e:
        raise RuntimeError(f"[LV3-PREPROCESS] Failed to copy /root/my_repo1 from container: {e}")

    # 4. 把本地 test/fp2 file 替换到本地 test/repo_name 下的正确位置
    if os.path.exists(fp2_file_path):
        target_repo_path = os.path.join(test_host_path, repo_name)
        # 处理 fp2_relative_path，去掉开头的 repo_name 部分
        # 去掉开头的 "repo_name/" 部分
        relative_path_in_target_repo = fp2_relative_path[len(repo_name) + 1:]
        target_fp2_path = os.path.join(target_repo_path, relative_path_in_target_repo)

        if os.path.exists(target_repo_path):
            # 确保目标目录存在
            target_fp2_dir = os.path.dirname(target_fp2_path)
            os.makedirs(target_fp2_dir, exist_ok=True)

            shutil.copy2(fp2_file_path, target_fp2_path)
            logger.info(f"[LV3-PREPROCESS] Replaced fp2 file in repository: {target_fp2_path}")
        else:
            logger.error(f"[LV3-PREPROCESS] Target repository directory not found: {target_repo_path}")
            raise RuntimeError(f"[LV3-PREPROCESS] Target repository directory not found: {target_repo_path}")
    else:
        logger.warning("[LV3-PREPROCESS] fp2 file not found, skipping replacement")

    logger.info("[LV3-PREPROCESS] Repository preprocessing completed successfully")

def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
    pre_installed: bool = False,
    args: argparse.Namespace = None,
):
    """
    为每个 instance 初始化 runtime。这个 instance 的基镜像已经有了, 现在要做的就是把这个 instance 的
    prompt, condition 挂进 runtime 里。
    """
    logger.info('-' * 50)
    logger.info('BEGIN Runtime Initialization for ProgrammerBench')

    # 创建并进入 /WORK_DIR, 这是 agent 接受任务的地方
    action = CmdRunAction(command=f'mkdir -p /{WORK_DIR} && cd /{WORK_DIR}')
    obs = runtime.run_action(action)
    assert obs.exit_code == 0, f'Failed to create {WORK_DIR}: {obs.content}'
    logger.info(f'App created at /{WORK_DIR}')

    # 复制宿主机 task 文件夹到 runtime 中
    task_host_path = instance.get('task_path')
    test_host_path = instance.get('test_path')

    # 从 task_host_path 的父目录下的 config.yaml 中提取 task_level 和 repo_name 两个字段
    config_path = os.path.join(os.path.dirname(task_host_path), 'config.yaml')
    task_level = None
    repo_name = None

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            task_level = config_data.get('task_level', None)
            repo_name = config_data.get('repo_name', None)

            if task_level is None:
                logger.warning(f"task_level not found in config.yaml at '{config_path}'")
            if repo_name is None:
                logger.warning(f"repo_name not found in config.yaml at '{config_path}'")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file '{config_path}': {e}")
        except Exception as e:
            logger.error(f"Error reading config file '{config_path}': {e}")
    else:
        logger.warning(f"Config file not found at '{config_path}'")

    # 判断是否白盒测试
    is_white = args and hasattr(args, 'white') and args.white
    # 记录要临时拷贝进 task 的测试文件名
    whitebox_test_file = None

    if task_host_path and os.path.exists(task_host_path):
        # 如果是白盒测试，先把 test 目录下的测试文件拷贝到 task 目录
        if is_white and test_host_path and os.path.exists(test_host_path):
            path2test_file = os.path.join(test_host_path, 'path2test.txt')
            if os.path.exists(path2test_file):
                with open(path2test_file, 'r', encoding='utf-8') as f:
                    test_file_lines = [line.strip() for line in f if line.strip()]
                if test_file_lines:
                    # 只取第一行
                    test_file_name = os.path.basename(test_file_lines[0])
                    src_test_file = os.path.join(test_host_path, test_file_name)
                    dst_test_file = os.path.join(task_host_path, test_file_name)
                    if os.path.exists(src_test_file):
                        shutil.copy(src_test_file, dst_test_file)
                        whitebox_test_file = test_file_name
                        # 将 test_file_name 传递到 args 以便 prompt_modifications 使用
                        if args is not None:
                            setattr(args, 'whitebox_test_file', test_file_name)
                        logger.info(f"[WHITE] Copied test file '{src_test_file}' to '{dst_test_file}' for white-box testing.")
                    else:
                        # 说明有可能遇到 lv3 了

                        # 取第一行作为真正的 fp2 文件相对路径
                        fp2_relative_path = test_file_lines[0]
                        fp2_filename = os.path.basename(fp2_relative_path)
                        logger.info(f"Found fp2 file path from path2test.txt: {fp2_relative_path}")

                        # 0. 检查本地是否有 test/fp2_filename，若有，跳转到 step 2，若没有，进 step 1
                        fp2_file_path = os.path.join(test_host_path, fp2_filename)
                        logger.info("fp2 file not found, extracting from repo_name")

                        # 1. 把本地 test/repo_name 下的 fp2 file 找出来，copy 到 test/ 下
                        repo_path = os.path.join(test_host_path, repo_name)
                        if os.path.exists(repo_path):
                            # fp2_relative_path 形如 "trl/tests/test_best_of_n_sampler.py"
                            # 我们需要去掉开头的 repo_name 部分，因为我们已经在 test/repo_name 目录下了
                            relative_path_in_repo = fp2_relative_path[len(repo_name) + 1:]
                            repo_fp2_path = os.path.join(repo_path, relative_path_in_repo)

                            if os.path.exists(repo_fp2_path):
                                shutil.copy2(repo_fp2_path, fp2_file_path)
                                logger.info(f"Copied fp2 file from {repo_fp2_path} to {fp2_file_path}")
                            else:
                                logger.warning(f"fp2 file not found in {repo_fp2_path}")
                        else:
                            logger.warning(f"Repository directory not found: {repo_path}")
                        shutil.copy(src_test_file, dst_test_file)
                        whitebox_test_file = test_file_name
                        # 将 test_file_name 传递到 args 以便 prompt_modifications 使用
                        if args is not None:
                            setattr(args, 'whitebox_test_file', test_file_name)
                        logger.info(f"[WHITE] Copied test file '{src_test_file}' to '{dst_test_file}' for white-box testing.")

                        raise ValueError(f"[WHITE] Test file '{src_test_file}' not found for white-box testing.")
                else:
                    raise ValueError(f"[WHITE] path2test.txt is empty at '{path2test_file}'")
            else:
                raise ValueError(f"[WHITE] path2test.txt not found at '{path2test_file}'")

        # 根据消融变量配置修改 prompt.md
        original_prompt_path = None
        temp_prompt_path = None

        # 根据消融变量配置拿到 upd 内容
        prompt_modifications = get_prompt_modifications(args)

        # 检查是否需要删除接口描述部分
        without_interface = args and hasattr(args, 'without_interface') and args.without_interface

        # 如果有修改需求（添加内容或删除接口描述），则处理 prompt.md
        if prompt_modifications or without_interface:
            prompt_file_path = os.path.join(task_host_path, 'prompt.md')
            if os.path.exists(prompt_file_path):
                try:
                    # 读取原始 prompt.md 内容
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_content = f.read()

                    new_prompt_content = prompt_content

                    # 如果开启了 --without-interface，删除 "## Test and Interface Descriptions" 以下的内容
                    if without_interface:
                        # 查找 "## Test and Interface Descriptions" 的位置
                        interface_pattern = r'## Test and Interface Descriptions'
                        interface_match = re.search(interface_pattern, new_prompt_content)

                        if interface_match:
                            # 删除从 "## Test and Interface Descriptions" 开始到文件末尾的所有内容
                            new_prompt_content = new_prompt_content[:interface_match.start()].rstrip() + '\n'
                            logger.info("Removed '## Test and Interface Descriptions' section and everything below from prompt.md")
                        else:
                            logger.warning("'## Test and Interface Descriptions' section not found in prompt.md")

                    # 如果有需要添加的内容
                    if prompt_modifications:
                        # 查找 "Your available resources are listed below:" 的位置
                        resources_pattern = r'Your available resources are listed below:'
                        resources_match = re.search(resources_pattern, new_prompt_content)

                        if resources_match:
                            # 组合所有需要添加的注释
                            additional_notes = ''.join(prompt_modifications)

                            # 在 "Your available resources are listed below:" 之前插入新内容
                            insert_pos = resources_match.start()
                            new_prompt_content = new_prompt_content[:insert_pos] + additional_notes + new_prompt_content[insert_pos:]

                            modifications_list = [mod.strip() for mod in prompt_modifications if mod.strip()]
                            logger.info(f"Successfully added {len(modifications_list)} additional notes to prompt.md")
                        else:
                            logger.warning("No 'Your available resources are listed below:' section found in prompt.md for adding modifications")

                    # 备份原始文件到临时文件夹
                    temp_backup_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', prefix='prompt_backup_')
                    temp_backup_file.write(prompt_content)  # 写入原始内容
                    temp_backup_file.close()
                    temp_prompt_path = temp_backup_file.name
                    original_prompt_path = prompt_file_path

                    # 写入修改后的内容到原文件
                    with open(prompt_file_path, 'w', encoding='utf-8') as f:
                        f.write(new_prompt_content)

                    logger.info("Successfully modified prompt.md")

                except Exception as e:
                    raise ValueError(f"Failed to modify prompt.md: {e}")
            else:
                raise FileNotFoundError(f"prompt.md not found at {prompt_file_path}")

        # 复制 task 文件夹到 agent 接收任务的地方
        sandbox_task_path = f'/{WORK_DIR}'
        runtime.copy_to(task_host_path, sandbox_task_path, recursive=True)
        logger.info(f"Successfully copied task directory from '{task_host_path}' to sandbox at '{sandbox_task_path}'")

        # 白盒测试：复制后删除 task 目录下的测试文件
        if is_white and whitebox_test_file:
            try:
                file_to_remove = os.path.join(task_host_path, whitebox_test_file)
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
                    logger.info(f"[WHITE] Removed test file '{file_to_remove}' from task directory after mounting.")
            except Exception as e:
                raise ValueError(f"[WHITE] Failed to remove test file '{file_to_remove}': {e}")

        # 恢复原始的 prompt.md 文件（如果有备份的话）
        if original_prompt_path and temp_prompt_path and os.path.exists(temp_prompt_path):
            try:
                # 从临时备份文件恢复原始内容
                with open(temp_prompt_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(original_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                # 清理临时备份文件
                os.unlink(temp_prompt_path)
                logger.info("Restored original prompt.md file and cleaned up backup")
            except Exception as e:
                logger.warning(f"Failed to restore original prompt.md: {e}")
                # 尝试清理临时文件
                try:
                    os.unlink(temp_prompt_path)
                except:
                    pass

    else:
        # 如果任务路径不存在或未提供，中断评估
        raise FileNotFoundError(f'Task path not provided or not found: {task_host_path}')


    # 处理 lv1 级别任务的特殊逻辑
    if task_level == 1 and repo_name:
        if pre_installed:
            # 预装环境：先初始化 testbed，删除 repo_name 文件夹，然后执行 change.py
            logger.info(f"[LV1-PRE] Processing level 1 task with pre-installed environment")

            # 1. 删除 /testbed 里所有内容并从 /root/my_repo1/ 复制
            init_testbed_cmd = 'rm -rf /testbed/* && cp -r /root/my_repo1/* /testbed/'
            action = CmdRunAction(command=init_testbed_cmd)
            obs = runtime.run_action(action)
            if obs.exit_code == 0:
                logger.info(f"[LV1-PRE] Successfully initialized /testbed from /root/my_repo1/")
            else:
                raise ValueError(f"[LV1-PRE] Failed to initialize /testbed: {obs.content}")

            # 2. 删除 /root 下所有非隐藏文件和文件夹
            remove_non_hidden_cmd = 'find /root -maxdepth 1 ! -name ".*" ! -path "/root" -exec rm -rf {} +'
            action = CmdRunAction(command=remove_non_hidden_cmd)
            obs = runtime.run_action(action)
            if obs.exit_code == 0:
                logger.info(f"[LV1-PRE] Successfully removed non-hidden files and directories from /root")
            else:
                raise ValueError(f"[LV1-PRE] Failed to remove non-hidden files and directories: {obs.content}")

            # 3. 执行 replace_with_masked_code.py 脚本来替换 /testbed 下的文件
            change_cmd = f'cd /{WORK_DIR}/task/ && python replace_with_masked_code.py /testbed'
            action = CmdRunAction(command=change_cmd)
            obs = runtime.run_action(action)
            if obs.exit_code == 0:
                logger.info(f"[LV1-PRE] Successfully executed replace_with_masked_code.py script")
                logger.info(f"{obs.content}")
            else:
                raise ValueError(f"[LV1-PRE] Failed to execute replace_with_masked_code.py script: {obs.content}")

        else:
            # 非预装环境：创建 testbed 文件夹，然后将 repo_name/* 移动到 /testbed 下
            logger.info(f"[LV1-BASE] Processing level 1 task with base environment")

            # 1. 创建 /testbed 目录
            create_testbed_cmd = 'mkdir -p /testbed'
            action = CmdRunAction(command=create_testbed_cmd)
            obs = runtime.run_action(action)
            if obs.exit_code == 0:
                logger.info(f"[LV1-BASE] Successfully created /testbed directory")
            else:
                raise ValueError(f"[LV1-BASE] Failed to create /testbed directory: {obs.content}")

            # 2. 移动文件
            move_cmd = f'mv /{WORK_DIR}/task/{repo_name}/* /testbed/'
            action = CmdRunAction(command=move_cmd)
            obs = runtime.run_action(action)
            if obs.exit_code == 0:
                logger.info(f"[LV1-BASE] Successfully moved /{WORK_DIR}/task/{repo_name}/* to /testbed/")
            else:
                raise ValueError(f"[LV1-BASE] Failed to move files to /testbed: {obs.content}")

        # 删除 notes 文件夹
        remove_notes_cmd = f'rm -rf /{WORK_DIR}/task/notes'
        action = CmdRunAction(command=remove_notes_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV1] Successfully removed /{WORK_DIR}/task/notes directory")
        else:
            logger.info(f"[LV1] notes directory may not exist or already removed: {obs.content}")

        # 刪除 origin 文件夹
        remove_origin_cmd = f'rm -rf /{WORK_DIR}/task/origin'
        action = CmdRunAction(command=remove_origin_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV1] Successfully removed /{WORK_DIR}/task/origin directory")
        else:
            logger.info(f"[LV1] origin directory may not exist or already removed: {obs.content}")

        # 删除 replace_with_masked_code.py 文件
        remove_script_cmd = f'rm -f /{WORK_DIR}/task/replace_with_masked_code.py'
        action = CmdRunAction(command=remove_script_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV1] Successfully removed /{WORK_DIR}/task/replace_with_masked_code.py file")
        else:
            logger.info(f"[LV1] replace_with_masked_code.py may not exist or already removed: {obs.content}")

        # 删除 repo_name 文件夹
        remove_repo_cmd = f'rm -rf /{WORK_DIR}/task/{repo_name}'
        action = CmdRunAction(command=remove_repo_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV1] Successfully removed /{WORK_DIR}/task/{repo_name} directory")
        else:
            logger.info(f"[LV1] repo_name directory may not exist or already removed: {obs.content}")

        # 删除 /testbed 下的 .git 文件夹
        remove_git_cmd = 'rm -rf /testbed/.git'
        action = CmdRunAction(command=remove_git_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info("Successfully removed /testbed/.git directory")
        else:
            raise ValueError("No /testbed/.git directory found or failed to remove it")

    # lv3 的逻辑实现
    else:
        if pre_installed == False:
            raise ValueError('lv3 必须要预装环境')
        # 预装环境：先初始化 testbed，删除 repo_name 文件夹
        logger.info(f"[LV3-PRE] Processing level 3 task with pre-installed environment")

        # 0. 进行仓库预处理
        preprocess_lv3_repository(runtime, test_host_path, repo_name, logger)

        # 1. 删除 /testbed 里所有内容
        init_testbed_cmd = 'rm -rf /testbed/*'
        action = CmdRunAction(command=init_testbed_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV3-PRE] Successfully clean /testbed")
        else:
            raise ValueError(f"[LV3-PRE] Failed to clean /testbed: {obs.content}")

        # 2. 删除 /root 下所有非隐藏文件和文件夹
        remove_non_hidden_cmd = 'find /root -maxdepth 1 ! -name ".*" ! -path "/root" -exec rm -rf {} +'
        action = CmdRunAction(command=remove_non_hidden_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV3-PRE] Successfully removed non-hidden files and directories from /root")
        else:
            raise ValueError(f"[LV3-PRE] Failed to remove non-hidden files and directories: {obs.content}")

        # 删除 notes 文件夹
        remove_notes_cmd = f'rm -rf /{WORK_DIR}/task/notes'
        action = CmdRunAction(command=remove_notes_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV3] Successfully removed /{WORK_DIR}/task/notes directory")
        else:
            logger.info(f"[LV3] notes directory may not exist or already removed: {obs.content}")

        # 刪除 origin 文件夹
        remove_origin_cmd = f'rm -rf /{WORK_DIR}/task/origin'
        action = CmdRunAction(command=remove_origin_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV3] Successfully removed /{WORK_DIR}/task/origin directory")
        else:
            logger.info(f"[LV3] origin directory may not exist or already removed: {obs.content}")

        # 删除 replace_with_masked_code.py 文件
        remove_script_cmd = f'rm -f /{WORK_DIR}/task/replace_with_masked_code.py'
        action = CmdRunAction(command=remove_script_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV3] Successfully removed /{WORK_DIR}/task/replace_with_masked_code.py file")
        else:
            logger.info(f"[LV3] replace_with_masked_code.py may not exist or already removed: {obs.content}")

        # 删除 repo_name 文件夹
        remove_repo_cmd = f'rm -rf /{WORK_DIR}/task/{repo_name}'
        action = CmdRunAction(command=remove_repo_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info(f"[LV3] Successfully removed /{WORK_DIR}/task/{repo_name} directory")
        else:
            logger.info(f"[LV3] repo_name directory may not exist or already removed: {obs.content}")

        # 删除 /testbed 下的 .git 文件夹
        remove_git_cmd = 'rm -rf /testbed/.git'
        action = CmdRunAction(command=remove_git_cmd)
        obs = runtime.run_action(action)
        if obs.exit_code == 0:
            logger.info("Successfully removed /testbed/.git directory")
        else:
            raise ValueError("No /testbed/.git directory found or failed to remove it")

    logger.info('END Runtime Initialization for ProgrammerBench')
    logger.info('-' * 50)

    return task_level, repo_name

def get_instruction() -> MessageAction:
    """
    为 ProgrammerBench 任务生成代理的初始指令。
    该指令引导代理从 prompt.md 文件中读取问题描述。
    """
    # ProgrammerBench 的核心指令始终是相同的：读取已放置在工作区中的提示文件。
    instruction = (
        f"Please read the task description and requirements from the `/{WORK_DIR}/task` directory. "
        "The `prompt.md` file in the task directory contains a detailed description of the task. "
        "Your goal is to generate a complete codebase based on these instructions."
        # "请使用 str_replace_editor 工具读取文件, 而不是 sed"
    )
    # 原有的评测是不让 agent 接触浏览器, 但是我就想做这个的消融, 所以我注释掉了
    # if RUN_WITH_BROWSING:
    #     instruction += (
    #         '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
    #     )
    return MessageAction(content=instruction)

def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
    task_level: str,
    repo_name: str,
) -> dict[str, any]:
    """
    在代理完成工作后，将 /WORK_DIR 目录从沙箱复制到宿主机。以及保存一些文本结果, return 一个 dict 回去
    """
    logger.info('-' * 50)
    logger.info('BEGIN Runtime Completion for ProgrammerBench')

    # 0. 解耦: 检查容器中的 /testbed 目录是否存在, 如果存在, 把内容移动到 agent_output 下
    testbed_exists = False
    check_testbed_action = CmdRunAction(command='test -d /testbed && echo "EXISTS" || echo "NOT_EXISTS"')
    obs = runtime.run_action(check_testbed_action)
    if obs.exit_code == 0 and "EXISTS" in obs.content:
        testbed_exists = True
        logger.info("Found /testbed directory in container")

        # 创建 agent_output 目录保存 testbed 内容
        agent_output_dir = os.path.join(metadata.eval_output_dir, 'agent_output')
        os.makedirs(agent_output_dir, exist_ok=True)

        # 在 agent_output 下为这个实例创建目录
        testbed_output_dir = os.path.join(agent_output_dir, instance.instance_id)
        if os.path.exists(testbed_output_dir):
            shutil.rmtree(testbed_output_dir)
        os.makedirs(testbed_output_dir, exist_ok=True)

        try:
            # 从容器中拉取 /testbed 目录
            testbed_archive_path = runtime.copy_from('/testbed')
            logger.info(f"Pulled /testbed archive from sandbox: {testbed_archive_path}")

            # 解压 testbed 内容到 agent_output 目录
            extracted = False
            try:
                if zipfile.is_zipfile(testbed_archive_path):
                    with zipfile.ZipFile(str(testbed_archive_path), 'r') as zf:
                        zf.extractall(path=testbed_output_dir)
                    logger.info("Testbed archive extracted as ZIP successfully.")
                    extracted = True
                else:
                    logger.warning("Testbed archive is not a valid ZIP file")
            except Exception as zip_e:
                logger.warning(f"ZIP extraction failed for testbed: {zip_e}")

            # 如果ZIP失败，尝试tarfile格式
            if not extracted:
                try:
                    if tarfile.is_tarfile(testbed_archive_path):
                        with tarfile.open(testbed_archive_path, 'r:*') as tf:
                            tf.extractall(path=testbed_output_dir)
                        logger.info("Testbed archive extracted as TAR successfully.")
                        extracted = True
                except Exception as tar_e:
                    logger.warning(f"TAR extraction failed for testbed: {tar_e}")

            if extracted:
                logger.info(f"Successfully saved /testbed contents to {testbed_output_dir}")
            else:
                logger.error("Failed to extract testbed archive")

        except Exception as e:
            logger.error(f"Failed to copy /testbed from sandbox: {e}")
    else:
        logger.info("/testbed directory not found in container")

    # 1. 使用封装的测试函数执行 pytest 测试
    test_result = run_pytest_and_evaluate(
        runtime=runtime,
        instance=instance,
        work_dir=WORK_DIR,
        test_time=TEST_TIME,
        logger=logger,
        task_level=task_level,
        repo_name=repo_name,
    )

    # 2. 把 runtime 里的 /WORK_DIR copy 到宿主机
    sandbox_app_path = f'/{WORK_DIR}'
    host_apps_dir = os.path.join(metadata.eval_output_dir, 'apps')
    os.makedirs(host_apps_dir, exist_ok=True)
    # 在 apps 下为这个实例创建一个单独目录
    instance_dir = os.path.join(host_apps_dir, instance.instance_id)
    # 如果已经存在，就先删除，保证干净
    if os.path.exists(instance_dir):
        shutil.rmtree(instance_dir)
    os.makedirs(instance_dir, exist_ok=True)
    try:
        # 1) 从容器中拉取压缩包，返回本地 Path，支持重试机制
        archive_path = None
        max_retries = 3

        for retry_attempt in range(max_retries):
            try:
                archive_path = runtime.copy_from(sandbox_app_path)
                logger.info(f"Pulled archive from sandbox (attempt {retry_attempt + 1}): {archive_path}")

                # 验证文件基本信息
                if not os.path.exists(archive_path):
                    raise FileNotFoundError(f"Archive file not found: {archive_path}")

                file_size = os.path.getsize(archive_path)
                logger.info(f"Archive file size: {file_size} bytes")

                if file_size == 0:
                    raise ValueError("Archive file is empty")

                # 如果文件大小正常，跳出重试循环
                if file_size > 0:
                    break

            except Exception as retry_e:
                logger.warning(f"Copy attempt {retry_attempt + 1} failed: {retry_e}")
                if retry_attempt == max_retries - 1:
                    raise
                time.sleep(1)  # 等待1秒后重试

        if archive_path is None:
            raise RuntimeError("Failed to copy archive after all retries")

        # 2) 尝试多种格式解压到 instance_dir
        extracted = False

        # 首先尝试ZIP格式
        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(str(archive_path), 'r') as zf:
                    zf.extractall(path=instance_dir)
                logger.info("Archive detected as ZIP, extracted successfully.")
                extracted = True
            else:
                logger.warning("File is not a valid ZIP archive")

                # 检查文件头部来诊断问题
                with open(archive_path, 'rb') as f:
                    header = f.read(16)
                logger.info(f"Archive file header (hex): {header.hex()}")
                logger.info(f"Archive file header (ascii): {header}")

        except zipfile.BadZipFile as zip_e:
            logger.warning(f"ZIP extraction failed: {zip_e}")
        except Exception as zip_e:
            logger.warning(f"Unexpected error during ZIP extraction: {zip_e}")

        # 如果ZIP失败，尝试tarfile格式
        if not extracted:
            try:
                if tarfile.is_tarfile(archive_path):
                    with tarfile.open(archive_path, 'r:*') as tf:
                        tf.extractall(path=instance_dir)
                    logger.info("Archive detected as TAR, extracted successfully.")
                    extracted = True
                else:
                    logger.warning("File is not a valid TAR archive")
            except Exception as tar_e:
                logger.warning(f"TAR extraction failed: {tar_e}")

        # 如果都失败了，尝试作为普通文件直接复制
        if not extracted:
            logger.warning("Attempting to copy as regular file...")
            try:
                dest_file = os.path.join(instance_dir, os.path.basename(archive_path))
                shutil.copy2(archive_path, dest_file)
                logger.info(f"Copied as regular file to: {dest_file}")
                extracted = True
            except Exception as copy_e:
                logger.error(f"Failed to copy as regular file: {copy_e}")

        if not extracted:
            # 生成详细的错误报告
            error_details = {
                'archive_path': str(archive_path),
                'file_size': os.path.getsize(archive_path) if os.path.exists(archive_path) else 0,
                'file_exists': os.path.exists(archive_path),
                'is_zipfile': zipfile.is_zipfile(archive_path) if os.path.exists(archive_path) else False,
            }

            try:
                error_details['is_tarfile'] = tarfile.is_tarfile(archive_path) if os.path.exists(archive_path) else False
            except:
                error_details['is_tarfile'] = False

            try:
                with open(archive_path, 'rb') as f:
                    error_details['file_header'] = f.read(32).hex()
            except:
                error_details['file_header'] = 'unable_to_read'

            raise RuntimeError(f"Unsupported archive format. Details: {error_details}")

    except Exception as e:
        error_msg = f"Failed to copy result from sandbox: {e}"
        logger.error(error_msg)
        logger.info('END Runtime Completion for ProgrammerBench')
        return {'status': 'failure', 'error': error_msg, 'instance_id': instance.instance_id, 'detail': {}}

    # 夹层: 如果测试执行失败，直接返回
    if test_result['status'] == 'failure':
        logger.info('END Runtime Completion for ProgrammerBench')
        return test_result
    detail = test_result.get('detail', {})

    # 3. 拷贝 test/repos 文件夹和 test/repo.json 到 metadata.eval_output_dir/repos/ 下
    repos_dir = os.path.join(metadata.eval_output_dir, 'repos')
    os.makedirs(repos_dir, exist_ok=True)

    # 3.1 拷贝 test/repos 文件夹到 repos/{instance_id}/
    test_repos_path = os.path.join(instance_dir, 'test', 'repos')
    if os.path.exists(test_repos_path):
        instance_repos_dir = os.path.join(repos_dir, instance.instance_id)
        if os.path.exists(instance_repos_dir):
            shutil.rmtree(instance_repos_dir)  # 如果已存在则先删除
        shutil.copytree(test_repos_path, instance_repos_dir)
        logger.info(f"Copied test/repos directory to {instance_repos_dir}")
    else:
        logger.warning(f"test/repos directory not found in {test_repos_path}")

    # 3.2 拷贝 test/repo.json 到 repos/{instance_id}.json
    repo_json_path = os.path.join(instance_dir, 'test', 'repo.json')
    if not os.path.exists(repo_json_path):
        error_msg = f"repo.json not found in {repo_json_path}"
        logger.error(error_msg)
        logger.info('END Runtime Completion for ProgrammerBench')
        return {'status': 'failure', 'error': error_msg, 'instance_id': instance.instance_id, 'detail': detail}
    # 拷贝并重命名
    instance_repo_json_path = os.path.join(repos_dir, f"{instance.instance_id}.json")
    shutil.copy(repo_json_path, instance_repo_json_path)
    logger.info(f"Copied repo.json to {instance_repo_json_path}")

    logger.info('END Runtime Completion for ProgrammerBench')
    return {
        'status': 'success',
        'error': None,
        'instance_id': instance.instance_id,
        'repo_json_path': instance_repo_json_path,
        'detail': detail,
    }

def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
    args: argparse.Namespace = None,  # 命令行参数，包含所有消融变量开关
) -> EvalOutput:
    # 从 args 中提取消融变量开关
    pre_installed = args.pre_installed if args and hasattr(args, 'pre_installed') else False
    gpu_devices = args.gpu if args and hasattr(args, 'gpu') else None
    enable_black_box = args.black if args and hasattr(args, 'black') else False  # 新增黑盒模式开关
    config = get_config(instance, metadata, pre_installed, gpu_devices, enable_black_box)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # Increase resource_factor with increasing attempt_id
    # 指数退避算法
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    # 拿到 runtime
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    # =================================================================
    # 定义并订阅结构化流式输出
    structured_log_dir = os.path.join(metadata.eval_output_dir, 'structured_logs')
    os.makedirs(structured_log_dir, exist_ok=True)
    structured_log_path = os.path.join(structured_log_dir, f'{instance.instance_id}.jsonl')
    if os.path.exists(structured_log_path):
        os.remove(structured_log_path)

    def structured_stream_subscriber(event: Event):
        """
        这个回调函数会在每个事件发生时被调用。
        它将事件序列化为 JSON 并写入特定于此实例的日志文件。
        """
        # 我们通常只关心 Agent 的行为和环境的反馈
        if event.source not in [EventSource.AGENT, EventSource.ENVIRONMENT]:
            return
        try:
            with open(structured_log_path, 'a') as f:
                data = event_to_dict(event)
                core = {k: v for k, v in data.items() if k != 'tool_call_metadata'}
                choice = {
                    'data': data,
                    'core': core,
                }
                f.write(json.dumps(choice[LOG_LEVEL], indent=2, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error writing to structured log for instance {instance.instance_id}: {e}")

    # 将我们的回调函数订阅到事件流
    callback_id = f'structured_log_{instance.instance_id}'
    runtime.event_stream.subscribe(
        EventStreamSubscriber.MAIN,
        structured_stream_subscriber,  # 回调函数
        callback_id                    # 回调 ID
    )
    logger.info(f"Structured event stream for instance {instance.instance_id} will be logged to: {structured_log_path}")
    # =================================================================

    try:
        # 对 runtime 进行初始化
        task_level, repo_name = initialize_runtime(runtime, instance, metadata, pre_installed, args)

        # 设置 CallTest 上下文，这样 calltest_aware_user_response 就能访问 runtime 和 instance
        set_calltest_context(runtime=runtime, instance=instance, work_dir=WORK_DIR, test_time=TEST_TIME,
                           task_level=task_level, repo_name=repo_name)

        # 拿到初始指令
        message_action = get_instruction()

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
            )
        )

        # if fatal error, throw EvalError to trigger re-run
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # 拿到结果: 包括 status, error 属性
        test_result = complete_runtime(runtime, instance, metadata, task_level, repo_name)
        logger.info(f"Completion result for instance {instance.instance_id}")
        # 把 test_result 写入输出目录下的 instance_status.txt 中
        instance_status_file = upsert_instance_status_jsonl(
            dir_path=metadata.eval_output_dir,
            record={
                "status": test_result.get("status"),
                "error": test_result.get("error"),
                "instance_id": instance.instance_id,
                "repo_json_path": test_result.get("repo_json_path"),
                "detail": test_result.get("detail"),
            },
            filename="instance_status.txt",   # 内容是 JSON Lines
            id_field="instance_id",
            use_lock=True,                     # 如未安装 filelock 也能运行，只是无锁
        )
        logger.info(f"Upserted instance status for {instance.instance_id} into {instance_status_file}")

    finally:
        runtime.close()


    # The test_result is already prepared above.
    if state is None:
        raise ValueError('State should not be None.')

    # this is NO LONGER the event stream, but an agent history that includes delegate agent's events
    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)
    instruction = message_action.content

    output = EvalOutput(
        instance_id=instance.instance_id,   # 实例 ID
        instruction=instruction,            # 最初指令
        test_result=test_result,            # 实验结果
        metadata=metadata,                  # metadata
        history=histories,                  # 历史事件
        metrics=metrics,                    # 资源消耗情况
        error=state.last_error if state and state.last_error else None,
    )
    return output

if __name__ == '__main__':
    parser = get_parser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with-browsing', action='store_true', help='启用上网能力')
    group.add_argument('--no-browsing', action='store_true', help='禁用上网能力')
    parser.add_argument(
        '--white',
        action='store_true',
        help='Enable white-box testing: copy test file into task directory before mounting, then remove after mounting.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='potatoQi-hf/ProgrammerBench',
        help='data set to evaluate on',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--skip-ids',
        nargs='+',
        default=[],
        help='A list of instance_ids to skip during evaluation. Cannot be used with --eval-ids.'
    )
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Absolute path to the ProgrammerBench directory (e.g. /home/user/ProgrammerBench)',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the latest timestamp directory instead of creating a new one. If no previous experiment exists, will create a new one.',
    )
    parser.add_argument(
        '--resume-timestamp',
        type=str,
        default=None,
        help='Resume from a specific timestamp directory (format: 2025-08-21/10-15-56). Takes precedence over --resume.',
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Report mode: Skip data processing and evaluation, only generate summary report from existing results.',
    )
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['huggingface', 'local'],
        default='huggingface',
        help='Data source: huggingface (load from HuggingFace Hub) or local (load from --path/config/metadata.csv)',
    )
    parser.add_argument(
        '--pre-installed',
        action='store_true',
        help='Use pre-installed environment image instead of base image (default: False)',
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default=None,
        help='Specify CUDA visible devices (e.g., 0 or 0,1). Leave empty to disable GPU.',
    )
    parser.add_argument(
        '--black',
        action='store_true',
        help='Enable black-box testing mode: enable CallTest tool for automated testing (default: False)',
    )
    parser.add_argument(
        '--without-interface',
        action='store_true',
        help='Remove content below "## Test and Interface Descriptions" from prompt (default: False)',
    )

    args, _ = parser.parse_known_args()

    # 优先使用命令行参数覆盖环境变量
    if 'RUN_WITH_BROWSING' not in globals():
        RUN_WITH_BROWSING = False
    if getattr(args, 'with_browsing', False):
        RUN_WITH_BROWSING = True
    elif getattr(args, 'no_browsing', False):
        RUN_WITH_BROWSING = False

    # 将 eval_ids 从逗号分隔的字符串转换为列表
    if args.eval_ids:
        args.eval_ids = [id.strip() for id in args.eval_ids.split(',') if id.strip()]
    else:
        args.eval_ids = []

    # 验证参数互斥性
    if args.skip_ids and args.eval_ids:
        raise ValueError("Cannot use both --skip-ids and --eval-ids simultaneously. Please use only one of them.")

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # 检查
    if not args.agent_cls:
        raise ValueError('Please specify the agent class with --agent_cls')

    # Get condenser config from environment variable
    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        # If no specific condenser config is provided via env var, default to NoOpCondenser
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    # 数据集名
    dataset_name = 'ProgrammerBench'

    # 根据数据源加载数据
    if args.data_source == 'local':
        # 从本地 CSV 文件读取数据
        metadata_path = os.path.join(args.path, 'config', 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Local metadata file not found: {metadata_path}")

        logger.info(f"Loading data from local file: {metadata_path}")
        dataset = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(dataset)} instances from local metadata file")
    else:
        # 从 HuggingFace Hub 加载数据 (默认行为)
        logger.info(f"Loading data from HuggingFace Hub: {DATASET_NAME}, split: {args.split}")
        hf_dataset = load_dataset(DATASET_NAME, split=args.split)
        dataset = pd.DataFrame(hf_dataset)
        logger.info(f"Loaded {len(dataset)} instances from HuggingFace Hub")

    # 拼接路径: 把 dataset['task_path'] 和 dataset['test_path'] 拼接成完整路径
    dataset['task_path'] = dataset['task_path'].apply(
        lambda x: os.path.join(args.path, x)
    )
    dataset['test_path'] = dataset['test_path'].apply(
        lambda x: os.path.join(args.path, x)
    )

    # 创建元数据
    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config=llm_config,                  # llm 配置
        dataset_name=dataset_description,
        agent_class=args.agent_cls,             # agent 的名字
        max_iterations=args.max_iterations,     # 最大执行次数
        eval_output_dir=args.eval_output_dir,   # 输出目录 (有默认的)
        condenser_config=condenser_config,      # 压缩器
        eval_note=args.eval_note,               # 评估备注
        details={},                             # 评估的额外信息
        mode='eval',                            # 模式 (自己加的一个参数)
        resume=args.resume or bool(args.resume_timestamp),  # resume 参数
        resume_timestamp=args.resume_timestamp,  # 具体的时间戳参数
    )

    # 输出文件路径
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')

    # 记录 resume 使用情况
    if args.resume_timestamp:
        logger.info(f"Resume mode enabled with specific timestamp: {args.resume_timestamp}")
    elif args.resume:
        logger.info("Resume mode enabled - using existing timestamp directory if available.")
    else:
        logger.info("Creating new timestamp directory for this evaluation run.")

    # 创建 instance_status.txt, 若存在则跳过
    instance_status_file = os.path.join(metadata.eval_output_dir, 'instance_status.txt')
    if not os.path.exists(instance_status_file):
        with open(instance_status_file, 'w') as f:
            pass
        logger.info(f"Created instance status file: {instance_status_file}")

    # 检查是否是报告模式
    if args.report:
        logger.info('Report mode enabled: Skipping data processing and evaluation, generating summary report only.')
        logger.info('Generating evaluation summary report...')
        generate_evaluation_summary_report(
            eval_output_dir=metadata.eval_output_dir,
            dataset=dataset,
            submitted_instances_count=None,  # 使用完整数据集的长度
            logger=logger
        )
        logger.info('Report generation completed successfully.')
        logger.info('You can find the report in the output directory: ' + metadata.eval_output_dir)
        exit(0)

    # 准备好实例 (prepare_dataset 会过滤一下, 比如做过的就不再做了)
    instances = prepare_dataset(
        dataset=dataset,
        output_file=output_file,
        eval_n_limit=args.eval_n_limit,
        eval_ids=args.eval_ids,
        skip_ids=args.skip_ids,
        mode='eval',    # 自己加的一个参数
    )

    # 使用 functools.partial 来传递 args 参数
    from functools import partial
    process_instance_with_args = partial(process_instance, args=args)

    # 运行评估过程
    # run_evaluation 会遍历 instances 的每一行, 将每一行作为参数传给 process_instance; metadata 会原封不动的传给 process_instance
    run_evaluation(
        dataset=instances,                         # 要评估的实例
        metadata=metadata,                         # 评估元数据
        output_file=output_file,                   # 输出文件路径
        num_workers=args.eval_num_workers,         # 并行工作进程数
        process_instance_func=process_instance_with_args,    # 处理单个实例的函数
        timeout_seconds=8*60*60,                   # 超时时间，8小时
        max_retries=5,                             # 最大重试次数
    )

    logger.info('Generating evaluation summary report...')
    generate_evaluation_summary_report(
        eval_output_dir=metadata.eval_output_dir,
        dataset=dataset,
        submitted_instances_count=None,  # 使用完整数据集的长度，而不是剩余实例数
        logger=logger
    )

    logger.info('Evaluation completed successfully.')
    logger.info('You can find the results in the output directory: ' + metadata.eval_output_dir)
