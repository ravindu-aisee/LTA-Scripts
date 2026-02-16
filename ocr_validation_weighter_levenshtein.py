"""
OCR Fuzzy String Matching using Weighted Levenshtein Distance
Singapore LED confusion matrix - matches Swift implementation
"""

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
REFERENCE_LIST = [
    "100",
    "11",
    "12",
    "13",
    "197",
    "2",
    "21",
    "26",
    "31",
    "32",
    "33",
    "51",
    "62",
    "63",
    "67",
    "7",
    "80",
    "853M",
]
SIMILARITY_THRESHOLD = 50.0

# Singapore LED confusion matrix costs (matches Swift implementation)
SUBSTITUTION_COSTS = {
    "5": {"S": 0.1, "6": 0.3},
    "S": {"5": 0.1},
    "0": {"O": 0.0, "D": 0.2, "U": 0.4},
    "O": {"0": 0.0},
    "8": {"B": 0.2, "0": 0.3},
    "B": {"8": 0.2},
    "1": {"I": 0.1, "7": 0.3, "T": 0.3},
    "I": {"1": 0.1},
    "7": {"1": 0.3},
    "T": {"1": 0.3},
    "2": {"Z": 0.2},
    "Z": {"2": 0.2},
    "4": {"A": 0.4},
    "A": {"4": 0.4},
    "M": {"N": 0.3},
    "N": {"M": 0.3},
}


class WeightedLevenshteinMatcher:
    """
    Weighted Levenshtein matcher using Singapore LED confusion matrix
    Matches Swift implementation exactly
    """

    def __init__(self, reference_list, similarity_threshold=55.0):
        """
        Initialize matcher

        Args:
            reference_list: List of valid strings to match against
            similarity_threshold: Minimum similarity percentage (0-100)
        """
        self.reference_list = reference_list
        self.normalized_reference_list = [self.normalize(ref) for ref in reference_list]
        self.similarity_threshold = max(0, min(100, similarity_threshold))

    @staticmethod
    def normalize(input_str):
        """
        Normalize string (uppercase and strip whitespace)

        Args:
            input_str: String to normalize

        Returns:
            Normalized string
        """
        return input_str.upper().strip()

    @staticmethod
    def substitution_cost(c1, c2):
        """
        Get substitution cost between two characters

        Args:
            c1: First character
            c2: Second character

        Returns:
            Cost value (0.0 = identical, 1.0 = completely different)
        """
        if c1 == c2:
            return 0.0

        # Check forward lookup
        if c1 in SUBSTITUTION_COSTS and c2 in SUBSTITUTION_COSTS[c1]:
            return SUBSTITUTION_COSTS[c1][c2]

        # Check reverse lookup
        if c2 in SUBSTITUTION_COSTS and c1 in SUBSTITUTION_COSTS[c2]:
            return SUBSTITUTION_COSTS[c2][c1]

        return 1.0

    def weighted_levenshtein_distance(self, s1, s2):
        """
        Calculate weighted Levenshtein distance between two strings

        Args:
            s1: First string
            s2: Second string

        Returns:
            Distance value (0 = identical strings)
        """
        len1 = len(s1)
        len2 = len(s2)

        if len1 == 0:
            return float(len2)
        if len2 == 0:
            return float(len1)

        # Create distance matrix
        matrix = [[0.0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

        # Initialize first column and row
        for i in range(len1 + 1):
            matrix[i][0] = float(i)
        for j in range(len2 + 1):
            matrix[0][j] = float(j)

        # Calculate distances
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                char1 = s1[i - 1]
                char2 = s2[j - 1]

                sub_cost = self.substitution_cost(char1, char2)

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1.0,  # deletion
                    matrix[i][j - 1] + 1.0,  # insertion
                    matrix[i - 1][j - 1] + sub_cost,  # substitution
                )

        return matrix[len1][len2]

    def similarity_ratio(self, s1, s2):
        """
        Calculate similarity ratio between two strings

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

        normalized_s1 = self.normalize(s1)
        normalized_s2 = self.normalize(s2)

        distance = self.weighted_levenshtein_distance(normalized_s1, normalized_s2)
        max_len = max(len(normalized_s1), len(normalized_s2))

        similarity = (1.0 - distance / float(max_len)) * 100.0
        return round(similarity, 2)

    def find_best_match(self, query):
        """
        Find the best matching string(s) from reference list

        Args:
            query: String to match

        Returns:
            List of dictionaries with best match info, or empty list if below threshold
        """
        normalized_query = self.normalize(query)

        if not normalized_query or not self.normalized_reference_list:
            return []

        best_matches = []
        best_score = 0.0

        for idx, normalized_reference in enumerate(self.normalized_reference_list):
            score = self.similarity_ratio(normalized_query, normalized_reference)

            if score > best_score:
                best_score = score
                best_matches = [
                    {"match": self.reference_list[idx], "score": score, "index": idx}
                ]
            elif score == best_score and score > 0:
                best_matches.append(
                    {"match": self.reference_list[idx], "score": score, "index": idx}
                )

        if best_score >= self.similarity_threshold:
            return best_matches

        return []

    def get_all_scores(self, query):
        """
        Get similarity scores for all references

        Args:
            query: String to match

        Returns:
            List of all scores, sorted by score descending
        """
        normalized_query = self.normalize(query)

        if not normalized_query or not self.normalized_reference_list:
            return []

        scores = []
        for idx, normalized_reference in enumerate(self.normalized_reference_list):
            score = self.similarity_ratio(normalized_query, normalized_reference)
            distance = self.weighted_levenshtein_distance(
                normalized_query, normalized_reference
            )
            scores.append(
                {
                    "reference": self.reference_list[idx],
                    "score": score,
                    "distance": distance,
                    "index": idx,
                }
            )

        return sorted(scores, key=lambda x: x["score"], reverse=True)

    def find_all_matches(self, query):
        """
        Find all matches above threshold

        Args:
            query: String to match

        Returns:
            List of matches sorted by score (highest first)
        """
        normalized_query = self.normalize(query)

        if not normalized_query or not self.normalized_reference_list:
            return []

        matches = []
        for idx, normalized_reference in enumerate(self.normalized_reference_list):
            score = self.similarity_ratio(normalized_query, normalized_reference)

            if score >= self.similarity_threshold:
                matches.append(
                    {"match": self.reference_list[idx], "score": score, "index": idx}
                )

        return sorted(matches, key=lambda x: x["score"], reverse=True)


# ============================================================================
# INTERACTIVE MODE
# ============================================================================


def main():
    """Interactive terminal mode"""

    print("-" * 70)
    print("WEIGHTED LEVENSHTEIN - SINGAPORE LED CONFUSION MATRIX")
    print("-" * 70)

    # Initialize matcher
    matcher = WeightedLevenshteinMatcher(REFERENCE_LIST, SIMILARITY_THRESHOLD)

    print(f"\nReference List: {REFERENCE_LIST}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}%")
    print(f"Confusion Matrix Pairs: {sum(len(v) for v in SUBSTITUTION_COSTS.values())}")
    print("\nType 'quit' or 'exit' to stop\n")

    while True:
        ocr_input = input("Enter OCR text: ").strip()

        if ocr_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not ocr_input:
            print("Please enter a valid input\n")
            continue

        print("-" * 70)

        # Get all similarity scores
        all_scores = matcher.get_all_scores(ocr_input)

        print(f"Similarity Scores for '{ocr_input}':")
        for item in all_scores:
            status = "✓" if item["score"] >= SIMILARITY_THRESHOLD else "✗"
            print(
                f"  {status} {item['reference']:6s} - {item['score']:6.2f}% "
                f"(Distance: {item['distance']:.2f}, Index: {item['index']})"
            )

        # Get best matches
        best_matches = matcher.find_best_match(ocr_input)

        print()
        if best_matches:
            if len(best_matches) == 1:
                print(
                    f"Best Match: {best_matches[0]['match']} "
                    f"(Score: {best_matches[0]['score']}%, Index: {best_matches[0]['index']})"
                )
            else:
                print(f"Best Matches (Tied with {best_matches[0]['score']}%):")
                for m in best_matches:
                    print(f"  - {m['match']} (Index: {m['index']})")
        else:
            print("A Bus is coming")

        # Show all matches above threshold
        all_matches = matcher.find_all_matches(ocr_input)

        if len(all_matches) > len(best_matches):
            print(f"\nAll Matches Above Threshold ({len(all_matches)}):")
            for m in all_matches:
                print(f"  - {m['match']} (Score: {m['score']}%, Index: {m['index']})")

        print()


if __name__ == "__main__":
    main()
