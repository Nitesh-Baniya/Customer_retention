"""Custom drift detection metrics for model monitoring."""

from __future__ import annotations

from typing import List, Optional

from evidently.core.datasets import Dataset
from evidently.core.metric_types import (
    BoundTest,
    SingleValueCalculation,
    SingleValueMetric,
)
from evidently.core.report import Context
from evidently.tests import lt


class BillingAmountMeanShift(SingleValueMetric):
    """Absolute difference between current and reference mean billing amounts."""

    column: str = "MonthlyCharges"
    threshold: float = 5.0

    def _default_tests_with_reference(
        self, context: Context
    ) -> List[BoundTest]:
        return [lt(self.threshold).bind_single(self.get_fingerprint())]


class BillingAmountMeanShiftCalculation(
    SingleValueCalculation[BillingAmountMeanShift]
):
    def calculate(
        self,
        context: Context,
        current_data: Dataset,
        reference_data: Optional[Dataset],
    ):
        if reference_data is None:
            raise ValueError(
                "BillingAmountMeanShift requires reference data"
            )
        target_column = self.metric.column
        current_mean = float(current_data.column(target_column).data.mean())
        reference_mean = float(
            reference_data.column(target_column).data.mean()
        )
        shift_value = abs(current_mean - reference_mean)
        result = self.result(shift_value)
        result.display_name = (
            f"|mean({target_column})_current - mean({target_column})_reference| "
            f"= {shift_value:.3f} (threshold={self.metric.threshold})"
        )
        return result

    def display_name(self) -> str:
        return (
            f"Absolute mean shift of '{self.metric.column}' "
            f"(current vs reference)"
        )
