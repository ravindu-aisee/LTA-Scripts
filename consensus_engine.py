from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum
import math
import time


class ConsensusResult(Enum):
    """Enum for consensus results"""
    NEW_LOCK = "new_lock"
    MAINTAINED = "maintained"
    DROPPED = "dropped"
    UNCERTAIN = "uncertain"


@dataclass
class ConsensusResultData:
    """Data class to hold consensus result with optional text and confidence"""
    result_type: ConsensusResult
    text: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Detection:
    """Represents a single detection with text, timestamp, and confidence"""
    text: Optional[str]
    timestamp: datetime
    confidence: float


class ConsensusEngine:
    """
    Smooths noisy OCR detections into a stable bus label.

    It keeps a short, time-decayed buffer of recent detections, scores each text
    by confidence and recency, and locks onto the top candidate once it crosses
    a threshold. A brief grace period prevents the lock from dropping during
    short gaps or weak frames, so the UI and feedback stay steady.

    Inputs: add_detection(text, confidence) per frame, where text may be None
    and confidence is the OCR confidence for that frame.
    Outputs: ConsensusResultData indicating a new lock, maintained lock, drop, or
    uncertain state.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        lock_threshold: float = 0.5,
        drop_threshold: float = 0.1,
        drop_grace_period: float = 0.6
    ):
        self.buffer_size = buffer_size
        self.lock_threshold = lock_threshold
        self.drop_threshold = drop_threshold
        self.drop_grace_period = drop_grace_period
        self.detection_buffer: List[Detection] = []
        self.current_locked_text: Optional[str] = None
        self.last_scores: Dict[str, float] = {}
        self.last_lock_sighting_time: Optional[datetime] = None

    def reset(self) -> None:
        """Reset the consensus engine to initial state"""
        self.detection_buffer.clear()
        self.current_locked_text = None
        self.last_scores.clear()
        self.last_lock_sighting_time = None

    def add_detection(self, text: Optional[str], confidence: float) -> ConsensusResultData:
        """
        Add a new detection and compute consensus result

        Args:
            text: Detected text (can be None)
            confidence: OCR confidence for this detection

        Returns:
            ConsensusResultData indicating the current consensus state
        """
        now = datetime.now()
        detection = Detection(text=text, timestamp=now, confidence=confidence)
        self.detection_buffer.append(detection)

        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)

        scores = self._compute_scores()
        self.last_scores = scores

        # --- Primary: lock onto top candidate if above lock_threshold ---
        if scores:
            top_candidate, score = max(scores.items(), key=lambda x: x[1])
            if score >= self.lock_threshold:
                if self.current_locked_text != top_candidate:
                    self.current_locked_text = top_candidate
                    self.last_lock_sighting_time = now
                    return ConsensusResultData(
                        result_type=ConsensusResult.NEW_LOCK,
                        text=top_candidate,
                        confidence=score
                    )

                self.last_lock_sighting_time = now
                return ConsensusResultData(
                    result_type=ConsensusResult.MAINTAINED,
                    text=top_candidate,
                    confidence=score
                )

        # --- Fallback: if we already have a lock, maintain it with grace/drop logic ---
        if self.current_locked_text is not None:
            locked_score = scores.get(self.current_locked_text, 0.0)
            if locked_score > 0.0:
                self.last_lock_sighting_time = now

            is_within_grace_period = False
            if self.last_lock_sighting_time is not None:
                time_diff = (now - self.last_lock_sighting_time).total_seconds()
                is_within_grace_period = time_diff <= self.drop_grace_period

            if locked_score >= self.drop_threshold or is_within_grace_period:
                return ConsensusResultData(
                    result_type=ConsensusResult.MAINTAINED,
                    text=self.current_locked_text,
                    confidence=locked_score
                )

            # Drop lock
            self.current_locked_text = None
            self.last_lock_sighting_time = None
            return ConsensusResultData(result_type=ConsensusResult.DROPPED)

        # No lock and nothing strong enough
        return ConsensusResultData(result_type=ConsensusResult.UNCERTAIN)

    def _compute_scores(self) -> Dict[str, float]:
        """
        Compute weighted scores for each detected text based on recency and confidence

        Returns:
            Dictionary mapping text to normalized scores
        """
        scores: Dict[str, float] = {}
        now = datetime.now()
        total_vote_weight = 0.0

        for detection in self.detection_buffer:
            if detection.text is None or not detection.text:
                continue

            age = (now - detection.timestamp).total_seconds()
            temporal_weight = math.exp(-age / 2.0)
            vote = float(temporal_weight * detection.confidence)

            scores[detection.text] = scores.get(detection.text, 0.0) + vote
            total_vote_weight += vote

        if total_vote_weight == 0:
            return {}

        return {text: score / total_vote_weight for text, score in scores.items()}

    def top_scores(self, limit: int) -> List[Tuple[str, float]]:
        """
        Get the top N scoring texts

        Args:
            limit: Maximum number of results to return

        Returns:
            List of (text, score) tuples sorted by score descending
        """
        if limit <= 0:
            return []

        return sorted(
            [(text, score) for text, score in self.last_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def get_current_state(self) -> Dict:
        """
        Get the current state of the consensus engine

        Returns:
            Dictionary with current locked text, scores, and buffer info
        """
        return {
            "locked_text": self.current_locked_text,
            "all_scores": self.last_scores,
            "top_scores": self.top_scores(5),
            "buffer_size": len(self.detection_buffer),
            "has_lock": self.current_locked_text is not None
        }


# -----------------------------------------------------------------------------
# NORMAL TEST IMPLEMENTATION
# -----------------------------------------------------------------------------
def run_tests():
    """Run various test scenarios with different OCR results"""

    test_scenarios = [
        {
            "name": "Scenario 1: Consistent detection (should lock on '52')",
            "detections": [
                ("52", 0.9), ("52", 0.85), ("52", 0.92), ("52", 0.88), ("52", 0.90),
                ("52", 0.87), ("52", 0.91), ("52", 0.89), ("52", 0.93), ("52", 0.86),
            ]
        },
        {
            "name": "Scenario 2: Detection with noise (52 with occasional S2)",
            "detections": [
                ("52", 0.9), ("52", 0.85), ("S2", 0.4), ("52", 0.88), ("52", 0.90),
                ("52", 0.87), ("S2", 0.3), ("52", 0.89), ("52", 0.93), ("52", 0.86),
            ]
        },
        {
            "name": "Scenario 3: Detection with gaps (None values)",
            "detections": [
                ("52", 0.9), ("52", 0.85), ("52", 0.92), (None, 0.0), (None, 0.0),
                ("52", 0.87), ("52", 0.91), (None, 0.0), ("52", 0.93), ("52", 0.86),
            ]
        },
        {
            "name": "Scenario 4: Value change (52 to 14)",
            "detections": [
                ("52", 0.9), ("52", 0.85), ("52", 0.92), ("14", 0.88), ("14", 0.90),
                ("14", 0.87), ("14", 0.91), ("14", 0.89), ("14", 0.93), ("14", 0.86),
            ]
        },
        {
            "name": "Scenario 5: Long gap (should drop lock)",
            "detections": [
                ("52", 0.9), ("52", 0.85), ("52", 0.92), (None, 0.0), (None, 0.0),
                (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0),
            ]
        },
        {
            "name": "Scenario 6: Low confidence detections",
            "detections": [
                ("52", 0.3), ("52", 0.2), ("14", 0.25), ("52", 0.35), ("14", 0.28),
                ("52", 0.22), ("14", 0.31), ("52", 0.26), ("14", 0.29), ("52", 0.24),
            ]
        },
        {
            "name": "Scenario 7: Mixed values with varying confidence",
            "detections": [
                ("A1", 0.5), ("B2", 0.6), ("A1", 0.7), ("C3", 0.4), ("A1", 0.8),
                ("A1", 0.75), ("B2", 0.5), ("A1", 0.82), ("A1", 0.79), ("A1", 0.85),
            ]
        },
    ]

    for scenario in test_scenarios:
        print(f"\n{'=' * 70}")
        print(scenario["name"])
        print(f"{'=' * 70}")

        engine = ConsensusEngine()
        last_result = None

        for i, (text, confidence) in enumerate(scenario["detections"], 1):
            result = engine.add_detection(text, confidence)
            last_result = result

            text_display = f"'{text}'" if text else "None"
            result_type = result.result_type.value.upper()

            if result.text is not None:
                conf_disp = f"{result.confidence:.3f}" if result.confidence is not None else "None"
                print(
                    f"Frame {i:2d}: Input: {text_display:6s} (conf: {confidence:.2f}) -> "
                    f"{result_type:12s} | Locked: '{result.text}' (score: {conf_disp})"
                )
            else:
                print(
                    f"Frame {i:2d}: Input: {text_display:6s} (conf: {confidence:.2f}) -> "
                    f"{result_type:12s}"
                )

            time.sleep(0.05)

        print(f"\n{'-' * 70}")
        print("FINAL OUTPUT SUMMARY:")
        print(f"{'-' * 70}")

        state = engine.get_current_state()
        if state["has_lock"]:
            print(f"LOCKED ON: '{state['locked_text']}'")
            print(f"Final confidence score: {state['all_scores'].get(state['locked_text'], 0.0):.3f}")
        else:
            print(f"NO LOCK (Status: {last_result.result_type.value.upper()})")

        print("\nAll candidate scores:")
        if state["all_scores"]:
            for t, s in sorted(state["all_scores"].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(s * 50)
                print(f"  '{t}': {s:.3f} {bar}")
        else:
            print("  (no scores)")

        print(f"\nBuffer contains {state['buffer_size']} detections")
        print()


# -----------------------------------------------------------------------------
# EDGE CASE TEST IMPLEMENTATION
# -----------------------------------------------------------------------------
def run_edge_case_tests():
    """Run aggressive edge case scenarios designed to challenge the consensus logic"""

    edge_case_scenarios = [
        {
            "name": "EDGE 1: Rapid alternation between two values (52 vs 14)",
            "description": "Tests if engine locks prematurely or flip-flops",
            "detections": [
                ("52", 0.9), ("14", 0.9), ("52", 0.9), ("14", 0.9), ("52", 0.9),
                ("14", 0.9), ("52", 0.9), ("14", 0.9), ("52", 0.9), ("14", 0.9),
            ]
        },
        {
            "name": "EDGE 2: Slow confidence decay (52 fading away)",
            "description": "Tests if engine drops lock appropriately as confidence degrades",
            "detections": [
                ("52", 0.95), ("52", 0.85), ("52", 0.70), ("52", 0.50), ("52", 0.30),
                ("52", 0.15), ("52", 0.08), ("52", 0.04), ("52", 0.02), ("52", 0.01),
            ]
        },
        {
            "name": "EDGE 3: High confidence noise attacking locked value",
            "description": "Tests if wrong value with high confidence can steal the lock",
            "detections": [
                ("52", 0.9), ("52", 0.9), ("52", 0.9), ("WRONG", 0.99), ("52", 0.9),
                ("52", 0.9), ("WRONG", 0.99), ("52", 0.9), ("52", 0.9), ("52", 0.9),
            ]
        },
        {
            "name": "EDGE 4: Empty strings and whitespace",
            "description": "Tests handling of empty/whitespace values",
            "detections": [
                ("52", 0.9), ("", 0.5), ("52", 0.9), (" ", 0.5), ("52", 0.9),
                ("  ", 0.5), ("52", 0.9), ("", 0.8), ("52", 0.9), ("52", 0.9),
            ]
        },
        {
            "name": "EDGE 5: Death by a thousand cuts (gradual value shift)",
            "description": "Tests if engine maintains lock during slow OCR drift",
            "detections": [
                ("52", 0.9), ("52", 0.9), ("52", 0.8), ("S2", 0.4), ("52", 0.6),
                ("S2", 0.6), ("52", 0.4), ("S2", 0.8), ("S2", 0.9), ("S2", 0.9),
            ]
        },
        {
            "name": "EDGE 6: All values are equally weak",
            "description": "Tests behavior when no clear winner exists",
            "detections": [
                ("A", 0.3), ("B", 0.3), ("C", 0.3), ("D", 0.3), ("E", 0.3),
                ("F", 0.3), ("G", 0.3), ("H", 0.3), ("I", 0.3), ("J", 0.3),
            ]
        },
        {
            "name": "EDGE 7: Late arrival of true value",
            "description": "Tests if correct value can overcome early wrong lock",
            "detections": [
                ("WRONG", 0.9), ("WRONG", 0.9), ("WRONG", 0.9), ("WRONG", 0.9), ("52", 0.95),
                ("52", 0.95), ("52", 0.95), ("52", 0.95), ("52", 0.95), ("52", 0.95),
            ]
        },
        {
            "name": "EDGE 8: Confidence just at threshold boundaries",
            "description": "Tests behavior at exact threshold values (lock=0.5, drop=0.1)",
            "detections": [
                ("52", 0.50), ("52", 0.50), ("52", 0.50), ("52", 0.10), ("52", 0.10),
                ("52", 0.09), ("52", 0.09), ("52", 0.11), ("52", 0.49), ("52", 0.51),
            ]
        },
        {
            "name": "EDGE 9: Zero confidence values",
            "description": "Tests handling of zero confidence detections",
            "detections": [
                ("52", 0.9), ("52", 0.9), ("52", 0.0), ("14", 0.0), ("52", 0.0),
                ("52", 0.9), ("52", 0.0), ("52", 0.0), ("52", 0.9), ("52", 0.9),
            ]
        },
        {
            "name": "EDGE 10: Adversarial pattern - cycling through many values",
            "description": "Tests if engine gets confused with many different values",
            "detections": [
                ("A", 0.7), ("B", 0.7), ("C", 0.7), ("D", 0.7), ("E", 0.7),
                ("A", 0.7), ("B", 0.7), ("C", 0.7), ("D", 0.7), ("E", 0.7),
            ]
        },
        {
            "name": "EDGE 11: Buffer overflow attack - same value after gap",
            "description": "Tests if old buffer values affect new lock incorrectly",
            "detections": [
                ("52", 0.9), ("52", 0.9), ("52", 0.9), (None, 0.0), (None, 0.0),
                (None, 0.0), (None, 0.0), ("14", 0.9), ("14", 0.9), ("14", 0.9),
            ]
        },
        {
            "name": "EDGE 12: Similar looking characters (OCR confusion)",
            "description": "Tests handling of visually similar characters",
            "detections": [
                ("O", 0.8), ("0", 0.8), ("O", 0.8), ("0", 0.8), ("O", 0.8),
                ("0", 0.8), ("O", 0.8), ("0", 0.8), ("O", 0.8), ("0", 0.8),
            ]
        },
        {
            "name": "EDGE 13: Grace period exploitation",
            "description": "Tests if grace period (0.6s) works correctly with timing",
            "detections": [
                ("52", 0.9), ("52", 0.9), ("52", 0.9), (None, 0.0), (None, 0.0),
                (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0),
            ]
        },
        {
            "name": "EDGE 14: Confidence explosion after lock",
            "description": "Tests if sudden high confidence can change locked value",
            "detections": [
                ("52", 0.6), ("52", 0.6), ("52", 0.6), ("14", 1.0), ("14", 1.0),
                ("14", 1.0), ("14", 1.0), ("52", 0.6), ("52", 0.6), ("52", 0.6),
            ]
        },
        {
            "name": "EDGE 15: All None values (complete OCR failure)",
            "description": "Tests behavior when OCR completely fails",
            "detections": [(None, 0.0)] * 10
        },
        {
            "name": "EDGE 16: Single frame lock attempt",
            "description": "Tests if single high-confidence frame can create lock",
            "detections": [
                (None, 0.0), (None, 0.0), ("52", 0.99), (None, 0.0), (None, 0.0),
                (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0),
            ]
        },
        {
            "name": "EDGE 17: Lock on wrong value, then correction flood",
            "description": "Tests recovery when initially locked on wrong value",
            "detections": [
                ("WRONG", 0.9), ("WRONG", 0.9), ("WRONG", 0.9), ("52", 0.95), ("52", 0.95),
                ("52", 0.95), ("52", 0.95), ("52", 0.95), ("52", 0.95), ("52", 0.95),
            ]
        },
        {
            "name": "EDGE 18: Negative confidence (invalid input)",
            "description": "Tests handling of invalid negative confidence",
            "detections": [
                ("52", 0.9), ("52", -0.5), ("52", 0.9), ("14", -1.0), ("52", 0.9),
                ("52", 0.9), ("52", 0.9), ("52", 0.9), ("52", 0.9), ("52", 0.9),
            ]
        },
        {
            "name": "EDGE 19: Confidence over 1.0 (invalid input)",
            "description": "Tests handling of invalid confidence > 1.0",
            "detections": [
                ("52", 0.9), ("52", 1.5), ("14", 2.0), ("52", 0.9), ("52", 10.0),
                ("52", 0.9), ("52", 0.9), ("52", 0.9), ("52", 0.9), ("52", 0.9),
            ]
        },
        {
            "name": "EDGE 20: Very long strings (potential memory issue)",
            "description": "Tests handling of unusually long detected strings",
            "detections": [
                ("52", 0.9), ("A" * 1000, 0.8), ("52", 0.9), ("B" * 5000, 0.8), ("52", 0.9),
                ("52", 0.9), ("52", 0.9), ("52", 0.9), ("52", 0.9), ("52", 0.9),
            ]
        },
    ]

    for scenario in edge_case_scenarios:
        print(f"\n{'=' * 70}")
        print(scenario["name"])
        print(f"Description: {scenario['description']}")
        print(f"{'=' * 70}")

        engine = ConsensusEngine()
        last_result = None

        for i, (text, confidence) in enumerate(scenario["detections"], 1):
            result = engine.add_detection(text, confidence)
            last_result = result

            text_display = (
                f"'{text[:20]}...'" if (text and len(text) > 20)
                else (f"'{text}'" if text else "None")
            )
            result_type = result.result_type.value.upper()

            if result.text is not None:
                locked_disp = f"'{result.text[:20]}...'" if len(result.text) > 20 else f"'{result.text}'"
                conf_disp = f"{result.confidence:.3f}" if result.confidence is not None else "None"
                print(
                    f"Frame {i:2d}: Input: {text_display:14s} (conf: {confidence:6.2f}) -> "
                    f"{result_type:12s} | Locked: {locked_disp} (score: {conf_disp})"
                )
            else:
                print(
                    f"Frame {i:2d}: Input: {text_display:14s} (conf: {confidence:6.2f}) -> "
                    f"{result_type:12s}"
                )

            time.sleep(0.05)

        print(f"\n{'-' * 70}")
        print("FINAL OUTPUT SUMMARY:")
        print(f"{'-' * 70}")

        state = engine.get_current_state()

        if state["has_lock"]:
            locked_text = state["locked_text"]
            locked_display = locked_text[:30] + "..." if len(locked_text) > 30 else locked_text
            print(f"LOCKED ON: '{locked_display}'")
            print(f"Final confidence score: {state['all_scores'].get(locked_text, 0.0):.3f}")
        else:
            print(f"NO LOCK (Status: {last_result.result_type.value.upper()})")

        print("\nTop candidate scores:")
        if state["all_scores"]:
            for t, s in sorted(state["all_scores"].items(), key=lambda x: x[1], reverse=True)[:5]:
                disp = t[:20] + "..." if len(t) > 20 else t
                bar = "█" * int(s * 50)
                print(f"  '{disp}': {s:.3f} {bar}")
        else:
            print("  (no scores)")

        print(f"\nBuffer contains {state['buffer_size']} detections\n")


if __name__ == "__main__":
    run_tests()
    run_edge_case_tests()
