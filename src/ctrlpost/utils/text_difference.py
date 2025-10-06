

def find_lcs_alignment(tokens1, tokens2):
    """Find longest common subsequence alignment between two token sequences."""
    m, n = len(tokens1), len(tokens2)

    # Create LCS table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    # Backtrack to find alignment
    alignment = []
    i, j = m, n

    while i > 0 and j > 0:
        if tokens1[i - 1] == tokens2[j - 1]:
            alignment.append(('match', i - 1, j - 1))
            i -= 1
            j -= 1
        elif lcs[i - 1][j] > lcs[i][j - 1]:
            alignment.append(('delete', i - 1, -1))
            i -= 1
        else:
            alignment.append(('insert', -1, j - 1))
            j -= 1

    # Handle remaining tokens
    while i > 0:
        alignment.append(('delete', i - 1, -1))
        i -= 1
    while j > 0:
        alignment.append(('insert', -1, j - 1))
        j -= 1

    alignment.reverse()
    return alignment


def find_differing_spans(tokens1, tokens2):
    """Find continuous spans of tokens that differ between two sequences."""
    alignment = find_lcs_alignment(tokens1, tokens2)

    spans1 = []  # Differing spans in tokens1
    spans2 = []  # Differing spans in tokens2

    current_span1 = []
    current_span2 = []

    for operation, idx1, idx2 in alignment:
        if operation == 'match':
            # End current spans if they exist
            if current_span1:
                spans1.append(current_span1)
                current_span1 = []
            if current_span2:
                spans2.append(current_span2)
                current_span2 = []
        elif operation == 'delete':
            current_span1.append(idx1)
        elif operation == 'insert':
            current_span2.append(idx2)

    # Add final spans if they exist
    if current_span1:
        spans1.append(current_span1)
    if current_span2:
        spans2.append(current_span2)

    return spans1, spans2


def annotate_tokens_with_spans(tokens, spans, marker_start='[F]', marker_end='[/F]'):
    """Annotate tokens by adding markers around differing spans."""
    if not spans:
        return ' '.join(tokens)

    result = []
    span_indices = set()

    # Flatten all span indices
    for span in spans:
        span_indices.update(span)

    i = 0
    while i < len(tokens):
        if i in span_indices:
            # Start of a span
            result.append(marker_start)
            # Add all consecutive tokens in this span
            span_start = i
            while i < len(tokens) and i in span_indices:
                result.append(tokens[i])
                i += 1
            result.append(marker_end)
        else:
            result.append(tokens[i])
            i += 1

    return result
