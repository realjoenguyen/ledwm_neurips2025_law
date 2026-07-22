import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def _calls_z_generator(node):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "ZGenerator"
        for child in ast.walk(node)
    )


def _assigns_self_attr_from_z_generator(node, attr_name):
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not (
            len(child.targets) == 1
            and isinstance(child.targets[0], ast.Attribute)
            and isinstance(child.targets[0].value, ast.Name)
            and child.targets[0].value.id == "self"
            and child.targets[0].attr == attr_name
        ):
            continue
        if (
            isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Name)
            and child.value.func.id == "ZGenerator"
        ):
            return True
    return False


def test_encoder_rssm_registers_obs_z_generator_during_init():
    tree = ast.parse((REPO_ROOT / "ledwm" / "nets" / "EncoderRSSM.py").read_text())

    init = _class_method(tree, "EncoderRSSM", "__init__")
    call = _class_method(tree, "EncoderRSSM", "__call__")

    assert _assigns_self_attr_from_z_generator(init, "obs_z")
    assert not _calls_z_generator(call)


def test_rssm_registers_prior_z_generator_during_init():
    tree = ast.parse((REPO_ROOT / "ledwm" / "RSSM.py").read_text())

    init = _class_method(tree, "RSSM", "__init__")
    prior = _class_method(tree, "RSSM", "_prior")

    assert _assigns_self_attr_from_z_generator(init, "prior_z_gen")
    assert not _calls_z_generator(prior)
