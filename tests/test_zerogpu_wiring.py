"""ZeroGPU startup requirement: decorated callback is directly bound to Gradio."""
import ast
from pathlib import Path


def test_direct_gradio_registration():
    app = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app.read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    gpu = fns["vlm_answer"]
    assert any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
               and d.func.attr == "GPU" for d in gpu.decorator_list)
    bindings = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in {"click", "then", "submit", "success"}]
    assert any(any(k.arg == "fn" and isinstance(k.value, ast.Name)
                   and k.value.id == "vlm_answer" for k in call.keywords)
               for call in bindings)
    binding = next(call for call in bindings if any(
        k.arg == "fn" and isinstance(k.value, ast.Name) and k.value.id == "vlm_answer"
        for k in call.keywords))
    assert binding.func.attr == "success"  # failed retrieval must never consume GPU quota
