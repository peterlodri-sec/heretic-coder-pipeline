import time

LABEL = "heretic-decensor"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DISK_GB = 300
OFFER_QUERY = "gpu_name=A100_SXM4 disk_space>=300"


class ProvisionError(RuntimeError):
    pass


def find_labeled_instance(vast, label: str = LABEL):
    for inst in vast.show_instances():
        if inst.get("label") == label:
            return inst
    return None


def start_instance(vast, instance_id, retries: int = 3, backoff: int = 60, poll_interval: int = 10):
    for attempt in range(1, retries + 1):
        vast.start_instance(id=instance_id)
        time.sleep(poll_interval)
        inst = vast.show_instance(id=instance_id)
        if inst.get("actual_status") == "running":
            return inst
        if attempt < retries:
            time.sleep(backoff)
    raise ProvisionError(f"instance {instance_id} did not reach running after {retries} attempts")


def rent_new_instance(vast, label: str = LABEL, query: str = OFFER_QUERY, image: str = IMAGE,
                       disk_gb: int = DISK_GB, poll_interval: int = 10, max_wait_polls: int = 30):
    offers = vast.search_offers(query=query)
    if not offers:
        raise ProvisionError(f"no offers matched query: {query}")
    offer = min(offers, key=lambda o: o["dph_total"])

    result = vast.create_instance(id=offer["id"], image=image, disk=disk_gb)
    instance_id = result["new_contract"]
    vast.label_instance(id=instance_id, label=label)

    for _ in range(max_wait_polls):
        inst = vast.show_instance(id=instance_id)
        if inst.get("actual_status") == "running":
            return inst
        time.sleep(poll_interval)
    raise ProvisionError(f"newly created instance {instance_id} did not reach running in time")


def provision(vast, label: str = LABEL):
    existing = find_labeled_instance(vast, label)
    if existing is None:
        return rent_new_instance(vast, label)
    if existing.get("actual_status") == "running":
        return existing
    # A labeled instance exists but is stopped: start it. Do NOT swallow a
    # start failure and fall through to renting — that masks a persistent API
    # error as "instance is gone" and orphans a second, billed instance.
    # Terminated instances drop out of show_instances entirely, so reaching
    # here means the instance is still ours to restart.
    return start_instance(vast, existing["id"])
