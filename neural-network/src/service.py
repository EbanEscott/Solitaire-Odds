"""
HTTP service that exposes the trained AlphaSolitaire policy–value network.

Run from the project root as:

    python -m src.service --checkpoint checkpoints/policy_value_latest.pt
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .action_encoding import ActionSpace, encode_action
from .log_loader import EpisodeStep
from .model import PolicyValueNet
from .state_encoding import encode_state


class _ModelBundle:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Support both old and new checkpoint formats
        state_dim = int(ckpt.get("feature_dim") or ckpt.get("state_dim", 532))
        num_actions = int(ckpt.get("action_space_size") or ckpt.get("num_actions", 238))
        index_to_action: List[str] = list(ckpt.get("action_index_map") or ckpt.get("index_to_action", []))

        self.action_space = ActionSpace(
            index_to_action=index_to_action,
            action_to_index={a: i for i, a in enumerate(index_to_action)},
        )

        model = PolicyValueNet(state_dim=state_dim, num_actions=num_actions)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        self.model = model
        self.device = device


def _pick_action_and_scores(
    bundle: _ModelBundle, request_payload: Dict[str, Any]
) -> Tuple[str, float, List[Tuple[str, float]]]:
    raw_step: Dict[str, Any] = {
        "tableau_visible": request_payload.get("tableau_visible")
        or request_payload.get("tableauVisible", []),
        "tableau_face_down": request_payload.get("tableau_face_down")
        or request_payload.get("tableauFaceDown", []),
        "foundation": request_payload.get("foundation", []),
        "talon": request_payload.get("talon", []),
        "stock_size": request_payload.get("stock_size")
        or request_payload.get("stockSize", 0),
        "legal_moves": request_payload.get("legal_moves")
        or request_payload.get("legalMoves", []),
        "recommended_moves": request_payload.get("recommended_moves", []),
        "chosen_command": request_payload.get("chosen_command", ""),
    }
    step = EpisodeStep(raw_step)

    state = encode_state(step).to(bundle.device).unsqueeze(0)
    
    # DEBUG: Print the state tensor to detect if it's always the same
    state_checksum = state.sum().item()
    print(
        f"[service] state_checksum={state_checksum:.6f} (talon={step.talon}, stock_size={step.stock_size})",
        flush=True,
    )

    with torch.no_grad():
        outputs = bundle.model(state)
        logits = outputs['policy']
        value_logits = outputs['value']
        probs = torch.softmax(logits[0], dim=-1)
        win_prob = torch.sigmoid(value_logits[0, 0]).item()

    legal_moves: List[str] = request_payload.get("legal_moves") or request_payload.get(
        "legalMoves", []
    )
    legal_scores: List[Tuple[str, float]] = []

    for move in legal_moves:
        idx = encode_action(bundle.action_space, move)
        if 0 <= idx < probs.shape[0]:
            legal_scores.append((move, float(probs[idx].item())))

    if legal_scores:
        chosen_command = max(legal_scores, key=lambda x: x[1])[0]
    else:
        best_idx = int(probs.argmax().item())
        chosen_command = bundle.action_space.index_to_action[best_idx]

    legal_scores.sort(key=lambda x: x[1], reverse=True)

    print(
        f"[service] chosen={chosen_command!r}, win_prob={win_prob:.3f}, "
        f"num_legal={len(legal_scores)}",
        flush=True,
    )

    return chosen_command, win_prob, legal_scores


class AlphaSolitaireHandler(BaseHTTPRequestHandler):
    model_bundle: _ModelBundle | None = None

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # type: ignore[override]
        if self.path != "/evaluate":
            self._send_json(404, {"error": "Not found"})
            return

        print("[service] received POST /evaluate", flush=True)
        print("[service] headers:\n", str(self.headers), flush=True)

        length_header = self.headers.get("Content-Length")
        if length_header is not None:
            try:
                length = int(length_header)
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return
            body = self.rfile.read(length)
        else:
            body = self.rfile.read()

        try:
            text = body.decode("utf-8")
            print("[service] raw body:", text, flush=True)
            payload = json.loads(text)
        except json.JSONDecodeError:
            print("[service] invalid JSON payload:", repr(body), flush=True)
            self._send_json(400, {"error": "Invalid JSON"})
            return

        if AlphaSolitaireHandler.model_bundle is None:
            self._send_json(500, {"error": "Model not initialised"})
            return

        try:
            chosen, win_prob, legal_scores = _pick_action_and_scores(
                AlphaSolitaireHandler.model_bundle, payload
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"Inference failed: {exc}"})
            return

        response = {
            "chosenCommand": chosen,
            "winProbability": win_prob,
            "legalMoves": [
                {"command": cmd, "probability": prob} for cmd, prob in legal_scores
            ],
        }
        self._send_json(200, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaSolitaire policy–value HTTP service")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/policy_value_latest.pt"),
        help="Path to the model checkpoint (.pt) file",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _ModelBundle(args.checkpoint, device=device)
    AlphaSolitaireHandler.model_bundle = bundle

    server = HTTPServer((args.host, args.port), AlphaSolitaireHandler)
    print(f"Serving AlphaSolitaire model on http://{args.host}:{args.port}/evaluate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()

