from __future__ import annotations

import warnings
from collections import UserString
from numbers import Number
from datetime import datetime

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal
    from pandas import Series


class VarType(UserString):
    """
    Prevent comparisons elsewhere in the library from using the wrong name.

    Errors are simple assertions because users should not be able to trigger
    them. If that changes, they should be more verbose.

    """
    # TODO VarType is an awfully overloaded name, but so is DataType ...
    # TODO adding unknown because we are using this in for scales, is that right?
    allowed = "numeric", "datetime", "categorical", "boolean", "unknown"

    def __init__(self, data):
        assert data in self.allowed, data
        super().__init__(data)

    def __eq__(self, other):
        assert other in self.allowed, other
        return self.data == other




def categorical_order(
        vector: Series,
        order: list | None = None
    ) -> list:
    """

        Return a list of unique data values using seaborn's ordering rules.

        Parameters
        ----------
        vector : Series
            Vector of "categorical" values
        order : list
            Desired order of category levels to override the order determined
            from the `data` object.

        Returns
        -------
        order : list
            Ordered list of category levels not including null values.


    """
    raise NotImplementedError('This function has been masked for testing')