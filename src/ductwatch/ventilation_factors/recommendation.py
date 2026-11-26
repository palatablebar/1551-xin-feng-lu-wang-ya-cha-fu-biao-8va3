from .field_retest import *  # noqa: F401,F403
from .field_retest import (  # noqa: F401
    MANUAL_RETEST,
    RetestAdvice,
    choose_field_retest as recommend_joint_options,
    recommend_field_retest,
)

recommend_joint_action = recommend_field_retest
build_retest_recommendation = recommend_field_retest
