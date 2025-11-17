from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast

import numpy as np
from pandas import DataFrame

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default

default = Default()




@dataclass
class Jitter(Move):
    """

        Random displacement along one or both axes to reduce overplotting.

        Parameters
        ----------
        width : float
            Magnitude of jitter, relative to mark width, along the orientation axis.
            If not provided, the default value will be 0 when `x` or `y` are set, otherwise
            there will be a small amount of jitter applied by default.
        x : float
            Magnitude of jitter, in data units, along the x axis.
        y : float
            Magnitude of jitter, in data units, along the y axis.

        Examples
        --------
        .. include:: ../docstrings/objects.Jitter.rst


    """

    width = "default"
    x = 0
    y = 0
    seed = None

    def __call__(
            self,
            data: DataFrame,
            groupby: GroupBy,
            orient: str,
            scales: dict[str, Scale]
        ) -> DataFrame:
        raise NotImplementedError('This function has been masked for testing')


@dataclass
class Dodge(Move):
    """

        Displacement and narrowing of overlapping marks along orientation axis.

        Parameters
        ----------
        empty : {'keep', 'drop', 'fill'}
        gap : float
            Size of gap between dodged marks.
        by : list of variable names
            Variables to apply the movement to, otherwise use all.

        Examples
        --------
        .. include:: ../docstrings/objects.Dodge.rst


    """

    empty = "keep"
    gap = 0
    by = None

    def __call__(
            self,
            data: DataFrame,
            groupby: GroupBy,
            orient: str,
            scales: dict[str, Scale]
        ) -> DataFrame:
        raise NotImplementedError('This function has been masked for testing')


@dataclass
class Stack(Move):
    """

        Displacement of overlapping bar or area marks along the value axis.

        Examples
        --------
        .. include:: ../docstrings/objects.Stack.rst


    """

    def _stack(self, df, orient):
        raise NotImplementedError('This function has been masked for testing')

    def __call__(
            self,
            data: DataFrame,
            groupby: GroupBy,
            orient: str,
            scales: dict[str, Scale]
        ) -> DataFrame:
        raise NotImplementedError('This function has been masked for testing')


@dataclass
class Shift(Move):
    """

        Displacement of all marks with the same magnitude / direction.

        Parameters
        ----------
        x, y : float
            Magnitude of shift, in data units, along each axis.

        Examples
        --------
        .. include:: ../docstrings/objects.Shift.rst


    """

    x = 0
    y = 0

    def __call__(
            self,
            data: DataFrame,
            groupby: GroupBy,
            orient: str,
            scales: dict[str, Scale]
        ) -> DataFrame:
        raise NotImplementedError('This function has been masked for testing')


@dataclass
class Norm(Move):
    """

        Divisive scaling on the value axis after aggregating within groups.

        Parameters
        ----------
        func : str or callable
            Function called on each group to define the comparison value.
        where : str
            Query string defining the subset used to define the comparison values.
        by : list of variables
            Variables used to define aggregation groups.
        percent : bool
            If True, multiply the result by 100.

        Examples
        --------
        .. include:: ../docstrings/objects.Norm.rst


    """

    func = "max"
    where = None
    by = None
    percent = False
    group_by_orient = False

    def _norm(self, df, var):
        raise NotImplementedError('This function has been masked for testing')

    def __call__(
            self,
            data: DataFrame,
            groupby: GroupBy,
            orient: str,
            scales: dict[str, Scale]
        ) -> DataFrame:
        raise NotImplementedError('This function has been masked for testing')


# TODO
# @dataclass
# class Ridge(Move):
#     ...