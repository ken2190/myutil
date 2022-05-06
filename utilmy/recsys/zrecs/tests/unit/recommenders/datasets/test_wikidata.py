# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest


from recommenders.datasets.wikidata import (
    search_wikidata,
    find_wikidata_id,
    query_entity_links,
    read_linked_entities,
    query_entity_description,
)


@pytest.fixture(scope="module")
def q():
    """function q.
    Doc::
            
            Args:
            Returns:
                
    """
    return {
        "correct": "the lord of the rings",
        "not_correct": "yXzCGhyFfWatQAPxeuRd09RqqWAMsCYRxZcxUDv",
        "entity_id": "Q15228",
    }


def test_find_wikidata_id(q):
    """function test_find_wikidata_id.
    Doc::
            
            Args:
                q:   
            Returns:
                
    """
    assert find_wikidata_id(q["correct"]) == "Q15228"
    assert find_wikidata_id(q["not_correct"]) == "entityNotFound"


@pytest.mark.skip(reason="Wikidata API is unstable")
def test_query_entity_links(q):
    """function test_query_entity_links.
    Doc::
            
            Args:
                q:   
            Returns:
                
    """
    resp = query_entity_links(q["entity_id"])
    assert "head" in resp
    assert "results" in resp


@pytest.mark.skip(reason="Wikidata API is unstable")
def test_read_linked_entities(q):
    """function test_read_linked_entities.
    Doc::
            
            Args:
                q:   
            Returns:
                
    """
    resp = query_entity_links(q["entity_id"])
    related_links = read_linked_entities(resp)
    assert len(related_links) > 5


@pytest.mark.skip(reason="Wikidata API is unstable")
def test_query_entity_description(q):
    """function test_query_entity_description.
    Doc::
            
            Args:
                q:   
            Returns:
                
    """
    desc = query_entity_description(q["entity_id"])
    assert desc == "1954–1955 fantasy novel by J. R. R. Tolkien"


def test_search_wikidata():
    """function test_search_wikidata.
    Doc::
            
            Args:
            Returns:
                
    """
    # TODO
    pass
