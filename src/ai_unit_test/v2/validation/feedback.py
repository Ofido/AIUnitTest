"""Feedback summarizer for AIUnitTest v2 retry loops."""

from ai_unit_test.v2.models import ValidationResult


class FeedbackSummarizer:
    """Summarize validation failures into structured feedback for retries."""

    def summarize(self, results: list[ValidationResult], attempt: int) -> list[str]:
        """Turn validation results into feedback strings for the next attempt."""
        feedback: list[str] = []

        feedback.append(f"Attempt {attempt} failed. The following validators reported issues:")

        for result in results:
            if not result.success:
                feedback.append(f"[{result.validator_name}] {result.summary}")
                if result.command_results:
                    for line in result.command_results[:10]:
                        feedback.append(f"  {line}")

        feedback.append("Please fix these issues in your next patch proposal.")
        return feedback

    def classify_failure(self, results: list[ValidationResult]) -> str:
        """Classify the type of failure for retry context."""
        for result in results:
            if not result.success:
                if result.validator_name == "syntax":
                    return "syntax_error"
                if result.validator_name == "pytest":
                    return "test_failure"
        return "unknown"
