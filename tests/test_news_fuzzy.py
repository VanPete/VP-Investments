from fetchers.news_fetcher import clean_company_name, fuzzy_match_articles


def test_clean_company_name_basic():
    assert clean_company_name("MegaCorp Inc.").lower() == "megacorp"


def test_fuzzy_match_selects_relevant():
    name = "Acme Corporation"
    arts = [
        {"title": "Acme wins contract", "description": "Good news for Acme"},
        {"title": "Unrelated topic", "description": "Nothing here"},
    ]
    matched = fuzzy_match_articles(name, arts, threshold=60)
    assert any("Acme wins" in a.get("title", "") for a in matched)
