import os
from features import pcap_to_graph_v13

def load_dataset_v13_openset(root_dir, logger, num_unknown_classes=5):
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    known_classes_names = subdirs[:-num_unknown_classes]
    unknown_classes_names = subdirs[-num_unknown_classes:]

    class_to_idx = {name: i for i, name in enumerate(known_classes_names)}
    for name in unknown_classes_names:
        class_to_idx[name] = -1

    logger.log(f"Loading Dataset: {len(known_classes_names)} Knowns, {len(unknown_classes_names)} Unknowns (Zero-days).")

    known_dataset, unknown_dataset = [], []

    for class_name in subdirs:
        class_dir = os.path.join(root_dir, class_name)
        label_int = class_to_idx[class_name]
        files = [f for f in os.listdir(class_dir) if f.endswith(".pcap") or f.endswith(".pcapng")]

        for f in files:
            data = pcap_to_graph_v13(os.path.join(class_dir, f), label_int)
            if data is not None and data.x.size(0) > 0:
                if label_int != -1:
                    known_dataset.append(data)
                else:
                    unknown_dataset.append(data)

    idx_to_class = {i: n for n, i in class_to_idx.items() if i != -1}
    return known_dataset, unknown_dataset, len(known_classes_names), idx_to_class