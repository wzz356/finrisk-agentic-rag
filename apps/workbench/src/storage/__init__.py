"""持久化适配器。"""

from storage.case_repository import CaseNotFoundError, CaseRepository, DuplicateCaseError

__all__ = ["CaseNotFoundError", "CaseRepository", "DuplicateCaseError"]
