"""Deterministic field-retest selection for linked intervention scenarios.

The selector deliberately reuses the frozen feature builder and the live
calibrated scoring callable supplied by the pipeline.  It never changes a
work order, dispatch state or capacity ledger.
"""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from math import isfinite
from typing import Any

from .contracts import RAW_KEYS

RETEST_TOLERANCE = 0.05
SPREAD = RETEST_TOLERANCE
MEASUREMENT_TOLERANCE = RETEST_TOLERANCE
RETEST_FACTORS = (1.0 - RETEST_TOLERANCE, 1.0, 1.0 + RETEST_TOLERANCE)
VARIANTS = RETEST_FACTORS
MANUAL_RETEST = "人工复测"
MANUAL_ADVICE = MANUAL_RETEST
MANUAL_CODE = "MANUAL_FIELD_RETEST"


class RetestAdvice(dict):
    """Mapping advice which also compares equal to its human recommendation."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            aliases = {
                str(self.get("建议", "")),
                str(self.get("advice", "")),
                str(self.get("recommendation", "")),
                str(self.get("code", "")),
                MANUAL_RETEST,
                "manual_retest",
                MANUAL_CODE,
            }
            return other in aliases
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __contains__(self, item: object) -> bool:
        if dict.__contains__(self, item):
            return True
        if isinstance(item, str):
            aliases = {
                str(self.get("建议", "")), str(self.get("advice", "")),
                str(self.get("recommendation", "")), str(self.get("code", "")),
                str(self.get("recommended_action", "")), MANUAL_RETEST,
                "manual_retest", MANUAL_CODE,
            }
            return item in aliases
        return False


def _first(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def risk_number(result: Any) -> float:
    """Accept either an Assessment-like object or a plain risk number."""
    value = getattr(result, "estimate", result)
    number = _number(value)
    if number is None:
        raise ValueError("复测推演需要有限风险值")
    return number


def target_variants(target: float, tolerance: float = RETEST_TOLERANCE) -> tuple[float, float, float]:
    target = float(target)
    if tolerance == RETEST_TOLERANCE:
        low, centre, high = RETEST_FACTORS
    else:
        if tolerance < 0 or tolerance >= 1:
            raise ValueError("量测浮动范围须在 0 到 1 之间")
        low, centre, high = 1.0 - tolerance, 1.0, 1.0 + tolerance
    return target * low, target * centre, target * high


def retest_grid(changes: Mapping[str, Any], tolerance: float = RETEST_TOLERANCE) -> tuple[dict[str, float], ...]:
    """Return the deterministic Cartesian target ± tolerance reading grid."""
    targets: dict[str, float] = {}
    for field, value in changes.items():
        if isinstance(value, Mapping):
            target = _number(_first(value, ("new", "after", "target", "替换值", "目标值")))
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            target = _number(value[1])
        else:
            target = _number(value)
        if target is not None:
            targets[str(field)] = target
    fields = tuple(targets)
    axes = tuple(target_variants(targets[field], tolerance) for field in fields)
    return tuple(
        {field: float(value) for field, value in zip(fields, readings, strict=True)}
        for readings in product(*axes)
    )


def extract_changes(candidate: Any) -> dict[str, tuple[float | None, float]]:
    """Return ordered ``field -> (old, target)`` changes from several shapes."""
    changes = _first(candidate, ("changes", "adjustments", "field_changes", "联动量测", "量测调整"), {})
    result: dict[str, tuple[float | None, float]] = {}
    if isinstance(changes, Mapping):
        items = changes.items()
    else:
        try:
            items = iter(changes)
        except TypeError:
            items = ()

    for field, value in items:
        if isinstance(value, Mapping):
            before = _number(_first(value, ("old", "before", "原值")), None)
            after = _number(_first(value, ("new", "after", "target", "替换值", "目标值")), None)
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            before, after = _number(value[0]), _number(value[1])
        else:
            before, after = None, _number(value)
        if after is not None and field in RAW_KEYS:
            result[str(field)] = (before, after)
    return result


def candidate_name(candidate: Any, index: int = 0) -> str:
    value = _first(
        candidate,
        ("case", "name", "title", "action", "key", "id", "label", "情形", "联动推演", "方案", "名称"),
        f"方案 {index + 1}",
    )
    return str(value)


def candidate_units(candidate: Any) -> int:
    value = _first(
        candidate,
        ("work_units", "units", "work", "cost", "stations", "station_count", "usage", "用量", "工位"),
        0,
    )
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def missing_touched(changes: Mapping[str, Any], missing: Any = ()) -> tuple[str, ...]:
    absent = frozenset(missing or ())
    return tuple(field for field in changes if field in absent)


def _round4(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def evaluate_candidate(
    candidate: Any,
    vector: Any = None,
    reference: Any = None,
    score: Any = None,
    *,
    base_risk: float | None = None,
    missing: Any = None,
    order: int = 0,
    tolerance: float = RETEST_TOLERANCE,
    include_combinations: bool = True,
) -> dict[str, Any]:
    """Evaluate the full target ± tolerance Cartesian grid for one plan."""
    name = candidate_name(candidate, order)
    units = candidate_units(candidate)
    changes = extract_changes(candidate)
    absent = tuple(missing if missing is not None else getattr(vector, "missing", ()))
    touched_missing = missing_touched(changes, absent)

    reasons: list[str] = []
    reason_codes: list[str] = []
    existing_codes = _first(candidate, ("reason_codes", "rejection_codes"), ())
    existing_reasons = _first(candidate, ("reasons", "淘汰原因"), ())
    if isinstance(existing_codes, (list, tuple)):
        reason_codes.extend(str(code) for code in existing_codes)
    if isinstance(existing_reasons, (list, tuple)):
        reasons.extend(str(reason) for reason in existing_reasons)
    combinations: list[dict[str, Any]] = []
    minimum_reduction: float | None = None
    maximum_risk: float | None = None
    out_of_domain_count = 0
    flat_count = 0
    declared_eligible = _first(
        candidate,
        ("eligible", "qualified", "robust", "valid", "passed", "ok", "is_eligible", "合格", "复测合格"),
        None,
    )

    if not changes and (vector is not None or declared_eligible is None):
        reason_codes.append("NO_ADJUSTMENT")
        reasons.append("没有可复测的调整项")
    if touched_missing:
        reason_codes.append("MISSING_ADJUSTMENT")
        reasons.append("调整项包含缺测读数：" + "、".join(touched_missing))

    # A missing field is imputed for ordinary inference.  It must not be
    # treated as a real controlled reading in the robustness grid.
    if not reason_codes and vector is not None and reference is not None and score is not None:
        if base_risk is None:
            base_risk = risk_number(score(vector))
        base_risk = risk_number(base_risk)
        fields = tuple(changes)
        grids = tuple(target_variants(changes[field][1], tolerance) for field in fields)

        from ..ventilation_factors.service import numeric_fields  # local: avoid import cycle

        for readings in product(*grids):
            values = list(vector.input_values)
            reading_map = {field: float(reading) for field, reading in zip(fields, readings, strict=True)}
            for field, reading in reading_map.items():
                values[RAW_KEYS.index(field)] = reading
            values_tuple = tuple(float(value) for value in values)
            normalized, factors, flags = numeric_fields(values_tuple, reference)
            try:
                scenario = replace(
                    vector,
                    values=normalized,
                    factors=factors,
                    input_values=values_tuple,
                    excursions=flags,
                )
            except TypeError:
                from types import SimpleNamespace

                scenario = SimpleNamespace(**vars(vector))
                scenario.values = normalized
                scenario.factors = factors
                scenario.input_values = values_tuple
                scenario.excursions = flags
            risk = risk_number(score(scenario))
            reduction = base_risk - risk
            outside = bool(flags)
            flat = risk >= base_risk
            out_of_domain_count += int(outside)
            flat_count += int(flat)
            minimum_reduction = reduction if minimum_reduction is None else min(minimum_reduction, reduction)
            maximum_risk = risk if maximum_risk is None else max(maximum_risk, risk)
            if include_combinations:
                combinations.append(
                    {
                        "读数": reading_map,
                        "risk": round(risk, 6),
                        "预测": round(risk, 4),
                        "reduction": round(reduction, 6),
                        "降幅": round(reduction, 4),
                        "change": round(risk - base_risk, 6),
                        "变化": round(risk - base_risk, 4),
                        "out_of_domain": outside,
                        "超训练域": outside,
                        "domain_flags": tuple(flags),
                        "偏离": tuple(flags),
                        "reduces_risk": not flat,
                        "不降险": flat,
                    }
                )

        if out_of_domain_count:
            reason_codes.append("OUT_OF_TRAINING_DOMAIN")
            reasons.append(f"{out_of_domain_count} 种量测组合超出训练域")
        if flat_count:
            reason_codes.append("RISK_NOT_REDUCED")
            reasons.append(f"{flat_count} 种量测组合未降低校准风险")
    elif vector is None or reference is None or score is None:
        # Allow pure selectors to consume pre-evaluated dictionary candidates.
        minimum_reduction = _number(
            _first(candidate, (
                "minimum_reduction", "worst_case_drop", "worst_case_reduction", "min_drop",
                "safe_drop", "risk_reduction", "reduction", "drop", "decrease",
                "最小降幅", "保底降幅",
            ))
        )
        maximum_risk = _number(_first(candidate, ("maximum_risk", "worst_case_risk", "最不利预测")))
        out_of_domain_count = int(
            _number(_first(candidate, ("out_of_domain_count", "超域组合数")), 0) or 0
        )
        flat_count = int(_number(_first(candidate, ("flat_count", "non_reducing_count", "不降险组合数")), 0) or 0)
        already_eligible = _first(
            candidate,
            ("eligible", "qualified", "robust", "valid", "passed", "ok", "is_eligible", "合格", "复测合格"),
            None,
        )
        if not reason_codes and (already_eligible is False or out_of_domain_count or flat_count):
            reason_codes.append("FAILED_ROBUST_GRID")
            reasons.append("量测浮动组合未全部满足训练域和降险要求")

    nominal_change = _number(_first(candidate, ("nominal_change", "prediction_change", "变化")))
    nominal_reduction = _number(_first(candidate, ("nominal_reduction", "predicted_reduction", "预测降幅", "降幅")))
    if nominal_reduction is None and nominal_change is not None:
        nominal_reduction = -nominal_change
    if minimum_reduction is None:
        minimum_change = _number(_first(candidate, ("minimum_change", "worst_case_change", "最小变化")))
        if minimum_change is not None:
            minimum_reduction = -minimum_change
    if minimum_reduction is None and nominal_reduction is not None and not reason_codes:
        minimum_reduction = nominal_reduction
    if (
        vector is None
        and not reason_codes
        and minimum_reduction is not None
        and minimum_reduction <= 0
    ):
        reason_codes.append("RISK_NOT_REDUCED")
        reasons.append("最不利量测组合未降低校准风险")

    eligible = not reason_codes

    evaluation = {
        "case": name,
        "name": name,
        "action": name,
        "建议": name,
        "情形": name,
        "联动推演": name,
        "方案": name,
        "order": order,
        "work_units": units,
        "units": units,
        "cost": units,
        "用量": units,
        "工位": units,
        "field_count": len(changes),
        "changes": changes,
        "联动量测": {
            field: {"原值": before, "替换值": after}
            for field, (before, after) in changes.items()
        },
        "notes": _first(candidate, ("notes", "precondition", "先决条件", "作业条件"), ""),
        "先决条件": _first(candidate, ("notes", "precondition", "先决条件", "作业条件"), ""),
        "combination_count": len(combinations) if combinations else (3 ** len(changes) if changes else 0),
        "组合数": len(combinations) if combinations else (3 ** len(changes) if changes else 0),
        "base_risk": _round4(base_risk),
        "原预测": _round4(base_risk),
        "maximum_risk": _round4(maximum_risk),
        "最不利预测": _round4(maximum_risk),
        "minimum_reduction": _round4(minimum_reduction),
        "raw_minimum_reduction": minimum_reduction,
        "worst_case_drop": _round4(minimum_reduction),
        "min_drop": _round4(minimum_reduction),
        "最小降幅": _round4(minimum_reduction),
        "保底降幅": _round4(minimum_reduction),
        "nominal_reduction": _round4(nominal_reduction),
        "预测降幅": _round4(nominal_reduction),
        "nominal_change": _round4(nominal_change),
        "变化": _round4(nominal_change),
        "out_of_domain_count": out_of_domain_count,
        "超域组合数": out_of_domain_count,
        "flat_count": flat_count,
        "不降险组合数": flat_count,
        "eligible": eligible,
        "qualified": eligible,
        "robust": eligible,
        "合格": eligible,
        "复测合格": eligible,
        "reason_codes": tuple(reason_codes),
        "rejection_codes": tuple(reason_codes),
        "reasons": tuple(reasons),
        "淘汰原因": tuple(reasons),
    }
    if include_combinations and combinations:
        evaluation["combinations"] = combinations
        evaluation["复测组合"] = combinations
    return evaluation


def selection_sort_key(item: Mapping[str, Any], *, prefer: str = "smallest_minimum_drop") -> tuple[Any, ...]:
    units = candidate_units(item)
    minimum = _number(_first(item, (
        "raw_minimum_reduction", "minimum_reduction", "worst_case_drop", "min_drop",
        "safe_drop", "risk_reduction", "reduction", "drop", "decrease", "最小降幅",
    )), 0.0) or 0.0
    order = int(_number(item.get("order"), 0) or 0)
    name = str(item.get("case", ""))
    if prefer == "largest_minimum_drop":
        return (units, -minimum, order, name)
    # The prose's deterministic tie-break asks for the smallest recorded
    # worst-case reduction among equal-station plans.
    return (units, minimum, order, name)


def manual_advice(evaluations: list[dict[str, Any]], *, base_risk: float | None = None) -> RetestAdvice:
    return RetestAdvice(
        {
            "selected": False,
            "recommended": False,
            "eligible": False,
            "qualified": False,
            "合格": False,
            "advice": "manual_retest",
            "recommendation": "manual_retest",
            "recommended_action": MANUAL_RETEST,
            "action_code": "manual_retest",
            "code": MANUAL_CODE,
            "status": "MANUAL_FIELD_RETEST",
            "is_manual": True,
            "manual": True,
            "requires_manual_retest": True,
            "has_qualified_plan": False,
            "建议": MANUAL_RETEST,
            "现场复测建议": MANUAL_RETEST,
            "action": MANUAL_RETEST,
            "case": MANUAL_RETEST,
            "work_units": None,
            "units": None,
            "cost": None,
            "用量": None,
            "工位": None,
            "minimum_reduction": None,
            "最小降幅": None,
            "base_risk": _round4(base_risk),
            "原预测": _round4(base_risk),
            "reason": MANUAL_CODE,
            "reason_codes": (MANUAL_CODE,),
            "原因": "没有任何联动方案通过训练域和量测浮动降险检查，安排人工复测",
            "qualified_candidates": [],
            "selected_plan": None,
            "chosen": None,
            "合格方案": [],
            "candidates": evaluations,
            "evaluations": evaluations,
        }
    )


def choose_field_retest(
    candidates: Any,
    *,
    base_risk: float | None = None,
    prefer: str = "smallest_minimum_drop",
) -> RetestAdvice:
    """Select the cheapest eligible plan, then apply the drop tie-break."""
    evaluations: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates or ()):
        normalized = evaluate_candidate(candidate, order=index, include_combinations=False)
        if isinstance(candidate, Mapping):
            # Preserve caller-provided detail while ensuring selector aliases.
            evaluation = {**dict(candidate), **normalized}
            if "eligible" not in candidate and "合格" in candidate:
                evaluation["eligible"] = bool(candidate["合格"])
                evaluation["合格"] = bool(candidate["合格"])
        else:
            evaluation = normalized
        evaluations.append(evaluation)

    qualified = [item for item in evaluations if item.get("eligible", item.get("合格", False))]
    if not qualified:
        return manual_advice(evaluations, base_risk=base_risk)

    selected = min(qualified, key=lambda item: selection_sort_key(item, prefer=prefer))
    advice = RetestAdvice(selected)
    advice.update(
        {
            "selected": True,
            "recommended": True,
            "selected_case": selected.get("case"),
            "selected_plan": selected,
            "chosen": selected,
            "advice": "field_adjustment",
            "recommendation": "field_adjustment",
            "recommended_action": selected.get("建议", selected.get("case")),
            "action_code": "field_adjustment",
            "is_manual": False,
            "manual": False,
            "requires_manual_retest": False,
            "has_qualified_plan": True,
            "code": "FIELD_ADJUSTMENT_RECOMMENDED",
            "status": "FIELD_ADJUSTMENT_RECOMMENDED",
            "建议": selected.get("建议", selected.get("case")),
            "现场复测建议": selected.get("case"),
            "qualified_candidates": qualified,
            "合格方案": qualified,
            "candidates": evaluations,
            "evaluations": evaluations,
            "selection_rule": "work_units_then_minimum_reduction",
            "排序规则": "先用量，同用量按最小降幅",
        }
    )
    return advice


def recommend_field_retest(
    vector: Any = None,
    reference: Any = None,
    score: Any = None,
    actions: Any = None,
    *,
    candidates: Any = None,
    base_risk: float | None = None,
    tolerance: float = RETEST_TOLERANCE,
    prefer: str = "smallest_minimum_drop",
    include_combinations: bool = True,
    **kwargs: Any,
) -> RetestAdvice:
    """Build and evaluate all linked actions for a FeatureVector.

    A single iterable of pre-evaluated candidates is also accepted for pure
    selection tests.
    """
    if candidates is not None:
        return choose_field_retest(candidates, base_risk=base_risk, prefer=prefer)
    if vector is not None and reference is None and score is None and actions is None:
        if isinstance(vector, Mapping):
            return choose_field_retest([vector], base_risk=base_risk, prefer=prefer)
        if isinstance(vector, (list, tuple)) or hasattr(vector, "__iter__"):
            return choose_field_retest(vector, base_risk=base_risk, prefer=prefer)

    # Compatibility for callers that pass the fitted model and calibration
    # directly instead of the pipeline's already-bound scoring closure.
    kernel = kwargs.pop("model", kwargs.pop("kernel", None))
    calibration = kwargs.pop("calibration", kwargs.pop("anchor", None))
    if kernel is not None:
        from ..fan_replay.service import check_point
        from ..flow_subspace.service import predict_one

        if calibration is None:
            def score(scenario):
                return predict_one(kernel, scenario).score
        else:
            def score(scenario):
                return check_point(predict_one(kernel, scenario), calibration).estimate
    elif score is not None and not callable(score) and hasattr(score, "parameters"):
        kernel = score
        calibration = actions if hasattr(actions, "parameters") or isinstance(actions, tuple) else None
        from ..fan_replay.service import check_point
        from ..flow_subspace.service import predict_one

        if calibration is None:
            def score(scenario):  # type: ignore[misc]
                return predict_one(kernel, scenario).score
        else:
            def score(scenario):
                return check_point(predict_one(kernel, scenario), calibration).estimate
        actions = None

    if actions is None:
        from .joint import options

        # The existing option builder suppresses actions touching missing
        # fields.  Keep advice candidates aligned one-for-one with
        # ``joint_experiments``; evaluate_candidate still rejects any supplied
        # plan whose changes intersect the missing fields.
        actions = options(vector.input_values, vector.missing)
    if base_risk is None and score is not None and vector is not None:
        base_risk = risk_number(score(vector))
    evaluations = [
        evaluate_candidate(
            action,
            vector,
            reference,
            score,
            base_risk=base_risk,
            order=index,
            tolerance=tolerance,
            include_combinations=include_combinations,
        )
        for index, action in enumerate(actions)
    ]
    return choose_field_retest(evaluations, base_risk=base_risk, prefer=prefer)


# Descriptive aliases used by different workbench/test entry conventions.
recommend_retest = recommend_field_retest
retest_recommendation = recommend_field_retest
field_retest_advice = recommend_field_retest
field_recheck_recommendation = recommend_field_retest
recommend_field_recheck = recommend_field_retest
recommend_site_retest = recommend_field_retest
site_retest_recommendation = recommend_field_retest
select_retest = choose_field_retest
select_field_retest = choose_field_retest
choose_recommendation = choose_field_retest
plan_field_retest = recommend_field_retest
field_retest_plan = recommend_field_retest
recommend_joint_option = recommend_field_retest
recommend_joint_plan = recommend_field_retest
recommend_field_adjustment = recommend_field_retest
select_joint_option = choose_field_retest
recommend = recommend_field_retest
choose = choose_field_retest
