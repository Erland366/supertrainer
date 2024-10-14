import pytest

from supertrainer import StrictDict

from ..utilities import print_in_test


class DummyObject:
    def __init__(self, value):
        self.value = value


def test_strictdict():
    sd = StrictDict()

    # Basic tests (unchanged)
    with pytest.raises(AttributeError):
        sd.test = 3
    sd.set_value("test", 3)
    assert sd.test == 3
    assert sd["test"] == 3
    with pytest.raises(KeyError):
        sd["test"] = 4

    # Test allow_modification (unchanged)
    with sd.allow_modification():
        sd["test2"] = 5
    assert sd["test2"] == 5
    assert sd.test2 == 5

    # Test non-serializable object (unchanged)
    with sd.allow_modification():
        sd.test3 = object()
    assert isinstance(sd.test3, object)

    # Test two levels of nesting with object
    obj1 = DummyObject("nested_object")
    sd.set_value("level1.object", obj1)
    assert sd.level1.object.value == "nested_object"
    assert isinstance(sd.level1.object, DummyObject)

    # Test three levels of nesting with object
    obj2 = DummyObject("deeply_nested_object")
    sd.set_value("deep.deeper.object", obj2)
    assert sd.deep.deeper.object.value == "deeply_nested_object"
    assert isinstance(sd.deep.deeper.object, DummyObject)

    # Test mixed nesting with objects and primitives
    sd.set_value("mixed.list.0", DummyObject("first_object"))
    sd.set_value("mixed.list.1", "second")

    assert isinstance(sd.mixed.list[0], DummyObject)
    assert sd.mixed.list[0].value == "first_object"
    assert sd.mixed.list[1] == "second"

    print_in_test(sd)

    # Test serialization
    dict_sd = sd.to_serializable_dict()
    assert "test3" not in dict_sd
    assert "level1" in dict_sd and "object" not in dict_sd["level1"]  # Object should be skipped
    assert (
        "deep" in dict_sd
        and "deeper" in dict_sd["deep"]
        and "object" not in dict_sd["deep"]["deeper"]
    )  # Object should be skipped
    assert "mixed" in dict_sd and "list" in dict_sd["mixed"]
    assert len(dict_sd["mixed"]["list"]) == 1  # Only the serializable "second" should be present
    assert (
        dict_sd["mixed"]["list"]["1"] == "second"
    )  # Need to convert to string since we convert it to string on the StrictDict
    print_in_test(dict_sd)

    # Test full dictionary conversion
    true_dict_sd = sd.to_dict()
    assert "test3" in true_dict_sd
    assert "level1" in true_dict_sd and "object" in true_dict_sd["level1"]
    assert isinstance(true_dict_sd["level1"]["object"], DummyObject)
    assert (
        "deep" in true_dict_sd
        and "deeper" in true_dict_sd["deep"]
        and "object" in true_dict_sd["deep"]["deeper"]
    )
    assert isinstance(true_dict_sd["deep"]["deeper"]["object"], DummyObject)
    assert "mixed" in true_dict_sd and "list" in true_dict_sd["mixed"]
    assert isinstance(true_dict_sd["mixed"]["list"]["0"], DummyObject)
    assert (
        true_dict_sd["mixed"]["list"]["1"] == "second"
    )  # Need to convert to string since we convert it to string on the StrictDict
    print_in_test(true_dict_sd)

    # Test that nested StrictDicts maintain their properties
    with pytest.raises(AttributeError):
        sd.level1.new_attr = "should fail"

    with pytest.raises(KeyError):
        sd["deep"]["deeper"]["new_key"] = "should also fail"

    # Test setting nested values on existing paths
    new_obj = DummyObject("new_nested_object")
    sd.set_value("level1.object.nested", new_obj)
    assert isinstance(sd.level1.object.nested, DummyObject)
    assert sd.level1.object.nested.value == "new_nested_object"

    # Test overwriting existing nested StrictDict with an object
    final_obj = DummyObject("final_object")
    sd.set_value("deep.deeper", final_obj)
    assert isinstance(sd.deep.deeper, DummyObject)
    assert sd.deep.deeper.value == "final_object"
    with pytest.raises(AttributeError):
        _ = sd.deep.deeper.object  # This should now fail as 'deeper' is no longer a StrictDict

    # test if list is not broken by that converting to string thingy
    sd.set_value("a.b.c.d.e.f", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sd.a.b.c.d.e.f == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sd.a.b.c.d.e.f[2] == 3

    # test serializable
    dict_sd = sd.to_serializable_dict()
    assert dict_sd["a"]["b"]["c"]["d"]["e"]["f"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print_in_test(true_dict_sd)
    # instantiate from dict
    sd = StrictDict(true_dict_sd)
    assert sd.level1.object.value == "nested_object"

    print("All tests passed!")
