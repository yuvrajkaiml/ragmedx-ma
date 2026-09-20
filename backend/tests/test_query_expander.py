from app.rag.retrievers.query_expander import QueryExpander


def main():
    print("=" * 80)
    print("Testing QueryExpander")
    print("=" * 80)

    expander = QueryExpander(max_queries=4)

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nOriginal query:")
    print(query)

    queries = expander.expand(query)

    print("\nExpanded queries:")
    print("-" * 80)

    for index, expanded_query in enumerate(queries, start=1):
        print(f"{index}. {expanded_query}")

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    assert len(queries) <= 4
    assert queries[0] == query

    assert any(
        "18F-FDG PET" in expanded
        for expanded in queries
    )

    assert any(
        "immune checkpoint inhibitors" in expanded
        for expanded in queries
    )

    assert len(
        {query.lower() for query in queries}
    ) == len(queries)

    # --------------------------------------------------------------
    # Empty query validation
    # --------------------------------------------------------------

    try:
        expander.expand("")

        raise AssertionError(
            "Empty query should raise ValueError."
        )

    except ValueError:
        pass

    print("\n" + "=" * 80)
    print("SUCCESS: Query expansion test completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()