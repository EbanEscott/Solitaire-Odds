"""
HTTP service that exposes the trained AlphaSolitaire policy–value network.

This is intended to be called from the Java AlphaSolitairePlayer over
localhost. It:

- Loads a trained checkpoint from `checkpoints/policy_value_latest.pt`.
- Reconstructs the action space used during training.
- Accepts JSON requests describing a Solitaire state and legal moves.
- Returns the chosen command, per-move probabilities for legal moves,
  and a win-probability estimate.

Example usage (from repo root):

    source .venv/bin/activate
    python3 service.py --checkpoint checkpoints/policy_value_latest.pt

Example request JSON (to POST /evaluate):

    {
      "tableau_visible": [["3♦","4♠"], ...],
      "tableau_face_down": [3,0,0,0,0,0,0],
      "foundation": [["A♣"], [], [], []],
      "talon": ["7♣"],
      "stock_size": 24,
      "legal_moves": ["turn", "move W T1", "move T1 4♠ F1"]
    }
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from action_encoding import ActionSpace, encode_action
from log_loader import EpisodeStep
from model import PolicyValueNet
from state_encoding import encode_state


class _ModelBundle:
    """Holds the model and action space for easy access in handlers."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        ckpt = torch.load(checkpoint_path, map_location=device)

        state_dim = int(ckpt["state_dim"])
        num_actions = int(ckpt["num_actions"])
        index_to_action: List[str] = list(ckpt["index_to_action"])

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
    """
    Run the model on the given request payload and choose an action.

    Returns:
        chosen_command, win_probability, [(move, probability), ...] for legal_moves
    """
    # Wrap the request payload into an EpisodeStep-like dict so we can reuse
    # encode_state without modifying it. Support both snake_case (internal)
    # and camelCase (JSON from Java) field names.
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

    with torch.no_grad():
        logits, value_logits = bundle.model(state)
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

    # If no legal moves were provided or none mapped correctly, fall back
    # to the global argmax over the whole action space.
    if legal_scores:
        chosen_command = max(legal_scores, key=lambda x: x[1])[0]
    else:
        best_idx = int(probs.argmax().item())
        chosen_command = bundle.action_space.index_to_action[best_idx]

    # Sort legal moves by descending probability for easier inspection.
    legal_scores.sort(key=lambda x: x[1], reverse=True)

    print(
        f"[service] chosen={chosen_command!r}, win_prob={win_prob:.3f}, "
        f"num_legal={len(legal_scores)}",
        flush=True,
    )

    return chosen_command, win_prob, legal_scores


class AlphaSolitaireHandler(BaseHTTPRequestHandler):
    """HTTP handler that exposes /evaluate for policy–value inference."""

    # This will be set externally before the server starts.
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
            # When Content-Length is not provided, read until EOF. With
            # 'Connection: close' from the client, this will return once
            # the request body has been sent.
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

        # Use camelCase field names to match the Java DTO.
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
