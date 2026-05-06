"""M2 memory_store.py 场景测试"""

import tempfile, os
from packages.core.src.memory_store import InMemoryMemoryStore, JSONFileMemoryStore
from packages.core.src.types import MemoryItem


def make_item(id: str, content: str = "test") -> MemoryItem:
    return MemoryItem(id=id, timestamp="D1", content=content, tags=["t"])


class TestInMemory:
    def test_add_and_retrieve(self):
        store = InMemoryMemoryStore()
        store.add("a", make_item("1"))
        store.add("a", make_item("2"))
        assert len(store.get_all("a")) == 2

    def test_get_recent(self):
        store = InMemoryMemoryStore()
        for i in range(10):
            store.add("a", make_item(str(i)))
        recent = store.get_recent("a", 3)
        assert len(recent) == 3
        assert recent[-1].id == "9"

    def test_get_recent_zero(self):
        store = InMemoryMemoryStore()
        store.add("a", make_item("1"))
        assert store.get_recent("a", 0) == []

    def test_empty_agent(self):
        store = InMemoryMemoryStore()
        assert store.get_all("nobody") == []

    def test_multiple_agents_isolated(self):
        store = InMemoryMemoryStore()
        store.add("a", make_item("1"))
        store.add("b", make_item("2"))
        assert len(store.get_all("a")) == 1
        assert len(store.get_all("b")) == 1


class TestJSONFile:
    def test_basic_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONFileMemoryStore(tmp)
            store.add("hero", make_item("1", "first"))
            store.add("hero", make_item("2", "second"))
            assert len(store.get_all("hero")) == 2
            assert store.get_recent("hero", 1)[0].id == "2"

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONFileMemoryStore(tmp)
            store.add("hero", make_item("1"))
            assert os.path.exists(os.path.join(tmp, "hero.json"))

    def test_cross_instance_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = JSONFileMemoryStore(tmp)
            s1.add("hero", make_item("1", "persist"))
            s2 = JSONFileMemoryStore(tmp)
            assert len(s2.get_all("hero")) == 1
            assert s2.get_all("hero")[0].content == "persist"

    def test_non_existent_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONFileMemoryStore(tmp)
            assert store.get_all("nobody") == []

    def test_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hero.json")
            with open(path, "w") as f:
                f.write("not json")
            store = JSONFileMemoryStore(tmp)
            assert store.get_all("hero") == []
