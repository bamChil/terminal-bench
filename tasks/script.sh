#!/bin/bash

# 最简单的版本，使用不同的 sed 分隔符
old_proxy="http://127.0.0.1:7890"
new_proxy="http://127.0.0.1:17890"

find . -name "Dockerfile" -type f | while read dockerfile; do
    if grep -q "ENV.*_proxy.*$old_proxy" "$dockerfile"; then
        echo "更新: $dockerfile"
        # 使用 # 作为分隔符
        sed -i.bak "s#ENV http_proxy=\"$old_proxy\"#ENV http_proxy=\"$new_proxy\"#g" "$dockerfile"
        sed -i.bak "s#ENV https_proxy=\"$old_proxy\"#ENV https_proxy=\"$new_proxy\"#g" "$dockerfile"
        sed -i.bak "s#ENV HTTP_PROXY=\"$old_proxy\"#ENV HTTP_PROXY=\"$new_proxy\"#g" "$dockerfile"
        sed -i.bak "s#ENV HTTPS_PROXY=\"$old_proxy\"#ENV HTTPS_PROXY=\"$new_proxy\"#g" "$dockerfile"
        # 清理备份文件
        rm -f "$dockerfile.bak"
    fi
done

echo "代理端口更新完成！"