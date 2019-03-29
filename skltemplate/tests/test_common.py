import pytest

from sklearn.utils.estimator_checks import check_estimator

from mnemultisubject import TemplateEstimator
from mnemultisubject import TemplateClassifier
from mnemultisubject import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
