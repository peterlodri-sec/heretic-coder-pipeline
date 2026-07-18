from unittest.mock import patch

import pytest

from shared.vast_provision import (
    LABEL,
    ProvisionError,
    find_labeled_instance,
    provision,
    rent_new_instance,
    start_instance,
)


class FakeVast:
    def __init__(self, instances=None, offers=None, start_results=None):
        self.instances = instances or []
        self.offers = offers or []
        self.start_results = start_results or {}
        self.started_ids = []
        self.created = []
        self.labeled = []

    def show_instances(self):
        return self.instances

    def show_instance(self, id):
        for inst in self.instances:
            if inst["id"] == id:
                return inst
        raise KeyError(id)

    def start_instance(self, id):
        self.started_ids.append(id)
        for inst in self.instances:
            if inst["id"] == id:
                inst["actual_status"] = self.start_results.get(id, "exited")

    def search_offers(self, query):
        return self.offers

    def create_instance(self, id, image, disk):
        new_id = 999
        self.created.append({"offer_id": id, "image": image, "disk": disk})
        self.instances.append({"id": new_id, "actual_status": "running",
                                "ssh_host": "ssh1.vast.ai", "ssh_port": 12345, "label": None})
        return {"new_contract": new_id}

    def label_instance(self, id, label):
        self.labeled.append((id, label))


def test_find_labeled_instance_returns_match():
    vast = FakeVast(instances=[{"id": 1, "label": "other"}, {"id": 2, "label": LABEL}])
    result = find_labeled_instance(vast)
    assert result["id"] == 2


def test_find_labeled_instance_returns_none_when_absent():
    vast = FakeVast(instances=[{"id": 1, "label": "other"}])
    assert find_labeled_instance(vast) is None


def test_start_instance_succeeds_on_first_try():
    vast = FakeVast(
        instances=[{"id": 5, "actual_status": "exited"}],
        start_results={5: "running"},
    )
    result = start_instance(vast, 5, retries=3, backoff=0, poll_interval=0)
    assert result["actual_status"] == "running"
    assert vast.started_ids == [5]


def test_start_instance_raises_after_exhausting_retries():
    vast = FakeVast(instances=[{"id": 5, "actual_status": "exited"}])
    with pytest.raises(ProvisionError):
        start_instance(vast, 5, retries=3, backoff=0, poll_interval=0)
    assert len(vast.started_ids) == 3


def test_rent_new_instance_picks_cheapest_offer_and_labels_it():
    vast = FakeVast(offers=[{"id": 10, "dph_total": 2.0}, {"id": 11, "dph_total": 1.0}])
    result = rent_new_instance(vast, poll_interval=0)
    assert result["actual_status"] == "running"
    assert vast.created[0]["offer_id"] == 11
    assert vast.labeled == [(999, LABEL)]


def test_rent_new_instance_raises_when_no_offers():
    vast = FakeVast(offers=[])
    with pytest.raises(ProvisionError):
        rent_new_instance(vast, poll_interval=0)


def test_provision_reuses_running_labeled_instance():
    vast = FakeVast(instances=[{"id": 2, "label": LABEL, "actual_status": "running"}])
    result = provision(vast)
    assert result["id"] == 2
    assert vast.started_ids == []
    assert vast.created == []


def test_provision_starts_stopped_labeled_instance():
    vast = FakeVast(
        instances=[{"id": 2, "label": LABEL, "actual_status": "exited"}],
        start_results={2: "running"},
    )
    # provision() calls start_instance() with its real default backoff/poll_interval
    # (60s/10s) — mock time.sleep so this test doesn't actually wait on those.
    with patch("shared.vast_provision.time.sleep"):
        result = provision(vast)
    assert result["id"] == 2
    assert vast.started_ids == [2]
    assert vast.created == []


def test_provision_raises_when_start_fails_and_does_not_rent():
    # A stopped labeled instance that won't start must NOT silently trigger a
    # second rent — that masks a persistent failure and orphans billing.
    vast = FakeVast(
        instances=[{"id": 2, "label": LABEL, "actual_status": "exited"}],
        offers=[{"id": 10, "dph_total": 1.0}],
    )
    with patch("shared.vast_provision.time.sleep"):
        with pytest.raises(ProvisionError):
            provision(vast)
    assert vast.created == []


def test_provision_rents_directly_when_no_labeled_instance():
    vast = FakeVast(offers=[{"id": 10, "dph_total": 1.0}])
    result = provision(vast)
    assert result["id"] == 999
