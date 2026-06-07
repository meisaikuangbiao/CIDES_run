from src.core.instruction_parser import (
    _split_sections,
    parse_instruction_offline,
)


SAMPLE_MD_1 = (
    "# Role\n你是美团外卖骑手的站长。\n\n"
    "# Task\n致电骑手通知合同。\n\n"
    "# Opening Line\n你好，请问是${rider_name}吗？我是站长。\n\n"
    "# Call Flow\n1. 告知合同生效。\n2. 提醒按时配送。\n\n"
    "# Knowledge Points (FAQ)\n- 单日合同：必须完成 **X 单**。\n- 多日合同：每天完成 **Y 单**。\n\n"
    "# Constraints\n- 每次回复控制在**约 30 个字以内**。\n"
    "- 如被问及超出职责范围的问题，回复：\"我向同事确认后再回电给你。\"\n"
    "- 保持语气随意，像打电话一样自然。\n"
    "- 避免重复回复。\n"
)


SAMPLE_MD_2 = (
    "# Role: Customer Support\n"
    "## Task: 通知机构客户。\n"
    "# Constraints:\n- 每次回复极简——最多15-20个字\n- 不能承诺给商家折扣券或优惠券\n- 不说\"好的\"、\"哈哈\"等语气词\n- 若商家说在开车，礼貌说\"那我稍后再打\"后挂断\n\n"
    "# Opening Line: 您好，请问您是负责人吗？\n"
    "# Conversation Flow:\n## Step 1: 身份确认\n- 若是负责人 → 进入第2步\n## Step 2: 确认是否知情\n## Step 3: 传达升级内容\n"
)


def test_split_sections_handles_inline_value():
    secs = _split_sections(SAMPLE_MD_2)
    assert secs["role"].startswith("Customer Support")
    assert "通知机构客户" in secs["task"]
    assert secs["opening_line"].startswith("您好")
    assert "Step 1" in secs["call_flow"]


def test_offline_parser_extracts_constraints_and_flow():
    spec = parse_instruction_offline("1", SAMPLE_MD_1)
    assert spec.role.startswith("你是美团")
    assert spec.task.startswith("致电骑手")
    assert spec.opening_line_template.startswith("你好")
    assert spec.constraints.hard.max_chars_per_reply == 30
    assert spec.constraints.hard.required_out_of_scope_reply
    assert spec.variables == ["rider_name"]
    assert len(spec.flow_nodes) == 2
    assert spec.flow_edges and spec.flow_edges[0].source == "S1"
    assert spec.knowledge and spec.knowledge[0].triggers


def test_offline_parser_recognises_step_headers():
    spec = parse_instruction_offline("2", SAMPLE_MD_2)
    assert spec.constraints.hard.no_discount_promise is True
    assert spec.constraints.hard.max_chars_per_reply == 20
    step_descs = [n.desc for n in spec.flow_nodes]
    assert "身份确认" in step_descs[0]
    assert any("挂断" in t for t in spec.constraints.termination)
