"""OCR fuzzy string matching workflow for bus service numbers.

This script exposes a simple interactive terminal tool that:

1. Accepts noisy OCR text for a bus service number as input.
2. Compares it against a fixed reference list of valid bus services
     using Levenshtein distance.
3. Converts the distance to a similarity score in percent (0–100).
4. Prints all similarity scores to help tune thresholds.
5. Applies threshold logic to interpret the OCR result:
     - If one or more services score at or above ``SIMILARITY_THRESHOLD``,
         they are treated as valid matches and the best match(es) are shown.
     - If no service reaches ``SIMILARITY_THRESHOLD`` but at least one
         score falls between ``LOWER_THRESHOLD`` and ``UPPER_THRESHOLD``,
         it reports that "a bus is coming" (low‑confidence indication).
     - If all scores are below ``LOWER_THRESHOLD``, it reports
         "No Match Found".

Configuration is controlled by:

* ``REFERENCE_LIST`` – list of valid bus service codes to match against.
* ``SIMILARITY_THRESHOLD`` – minimum similarity (in %) to treat a
    service as a positive match.
* ``LOWER_THRESHOLD`` / ``UPPER_THRESHOLD`` – band where results are
    considered weak but possibly meaningful, triggering the
    "a bus is coming" message.
"""

REFERENCE_LIST = ["65M", "123", "45T", "45C", "13", "122", "7", "77", "77M"]
SIMILARITY_THRESHOLD = 55.0
LOWER_THRESHOLD = 10.0
UPPER_THRESHOLD = 55.0


class StringMatcher:
    """
    String matching class using Levenshtein Distance algorithm
    """

    def __init__(self, reference_list, similarity_threshold):
        """
        Initialize matcher

        Args:
            reference_list: List of valid strings to match against
            similarity_threshold: Minimum similarity percentage (0-100)
        """
        self.reference_list = reference_list
        self.threshold = similarity_threshold

    def levenshtein_distance(self, s1, s2):
        """
        Calculate Levenshtein distance between two strings
        Number of single-character edits (insertions, deletions, substitutions)

        Args:
            s1: First string
            s2: Second string

        Returns:
            Integer distance value
        """
        len1 = len(s1)
        len2 = len(s2)

        # Create distance matrix
        matrix = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

        # Initialize first column and row
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Calculate distances
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        return matrix[len1][len2]

    def similarity_ratio(self, s1, s2):
        """
        Calculate similarity ratio between two strings (0-100)

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity percentage (0-100)
        """
        if not s1 and not s2:
            return 100.0
        if not s1 or not s2:
            return 0.0

        distance = self.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        # Convert distance to similarity percentage
        similarity = (1 - distance / max_len) * 100
        return round(similarity, 2)

    def find_best_match(self, query):
        """
        Find the best matching string(s) from reference list
        If multiple items have the same highest score, returns all of them

        Args:
            query: String to match

        Returns:
            List of dictionaries with best match info, or empty list if below threshold
            [
                {
                    'match': matched_string,
                    'score': similarity_score,
                    'index': index_in_reference_list
                },
                ...
            ]
        """
        if not query or not self.reference_list:
            return []

        best_matches = []
        best_score = 0.0

        for idx, reference in enumerate(self.reference_list):
            score = self.similarity_ratio(query, reference)

            if score > best_score:
                # Found a better score, reset the list
                best_score = score
                best_matches = [{"match": reference, "score": score, "index": idx}]
            elif score == best_score and score > 0:
                # Same score as current best, add to list
                best_matches.append({"match": reference, "score": score, "index": idx})

        # Check if best score meets threshold
        if best_score >= self.threshold:
            return best_matches

        return []

    def get_all_scores(self, query):
        """
        Get similarity scores for all references (for threshold tuning)

        Args:
            query: String to match

        Returns:
            List of dictionaries with all scores, sorted by score (highest first)
            [
                {'reference': string, 'score': float, 'index': int},
                ...
            ]
        """
        if not query or not self.reference_list:
            return []

        scores = []
        for idx, reference in enumerate(self.reference_list):
            score = self.similarity_ratio(query, reference)
            scores.append({"reference": reference, "score": score, "index": idx})

        # Sort by score (descending)
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    def find_all_matches(self, query):
        """
        Find all matches above threshold, sorted by score

        Args:
            query: String to match

        Returns:
            List of match dictionaries sorted by score (highest first)
        """
        if not query or not self.reference_list:
            return []

        matches = []
        for idx, reference in enumerate(self.reference_list):
            score = self.similarity_ratio(query, reference)

            if score >= self.threshold:
                matches.append({"match": reference, "score": score, "index": idx})

        # Sort by score (descending)
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches


# ============================================================================
# INTERACTIVE MODE
# ============================================================================


def main():
    """Interactive terminal mode"""

    print("-" * 70)
    print("OCR FUZZY MATCHING - LEVENSHTEIN DISTANCE")
    print("-" * 70)

    # Initialize matcher with global configuration
    matcher = StringMatcher(REFERENCE_LIST, SIMILARITY_THRESHOLD)

    print(f"\nReference List: {REFERENCE_LIST}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}%")
    print("\nType 'quit' or 'exit' to stop\n")

    while True:
        # Get input from user
        ocr_input = input("Enter OCR text: ").strip()

        # Check for exit commands
        if ocr_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not ocr_input:
            print("Please enter a valid input\n")
            continue

        print("-" * 70)

        # Get all similarity scores (for threshold tuning)
        all_scores = matcher.get_all_scores(ocr_input)

        print(f"Similarity Scores for '{ocr_input}':")
        for item in all_scores:
            status = "✓" if item["score"] >= SIMILARITY_THRESHOLD else "✗"
            print(
                f"  {status} {item['reference']:6s} - {item['score']:6.2f}% (Index: {item['index']})"
            )

        # Get best match above threshold
        best_matches = matcher.find_best_match(ocr_input)

        print()
        if best_matches:
            if len(best_matches) == 1:
                print(
                    f"Best Match: {best_matches[0]['match']} (Score: {best_matches[0]['score']}%, Index: {best_matches[0]['index']})"
                )
            else:
                print(f"Best Matches (Tied with {best_matches[0]['score']}%):")
                for m in best_matches:
                    print(f"  - {m['match']} (Index: {m['index']})")
        else:
            print(" A Bus is coming")
            # # Check if any score is between LOWER_THRESHOLD and UPPER_THRESHOLD
            # found_mid_score = False
            # for item in all_scores:
            #     if LOWER_THRESHOLD <= item["score"] < UPPER_THRESHOLD:
            #         found_mid_score = True
            #         break
            # if found_mid_score:
            #     print(" A Bus is coming")
            # else:
            #     print("No Match Found (all scores below threshold)")

        # Get all matches above threshold
        all_matches = matcher.find_all_matches(ocr_input)

        if len(all_matches) > len(best_matches):
            print(f"\nAll Matches Above Threshold ({len(all_matches)}):")
            for m in all_matches:
                print(f"  - {m['match']} (Score: {m['score']}%, Index: {m['index']})")

        print()


if __name__ == "__main__":
    main()
