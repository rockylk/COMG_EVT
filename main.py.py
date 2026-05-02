import os
import time
import argparse
from utils import Logger
from dataset import load_dataset_v13_openset
from trainer import train_v13_evt_v2

def main():
    parser = argparse.ArgumentParser(description="COMG-EVT Open-Set Traffic Recognition")
    parser.add_argument('--pcap_dir', type=str, required=True, 
                        help='Path to the PCAP dataset directory')
    parser.add_argument('--result_dir', type=str, default='./results', 
                        help='Directory to save logs and numpy arrays')
    parser.add_argument('--num_unknowns', type=int, default=5, 
                        help='Number of tail classes to treat as zero-day unknowns')

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    log_file = os.path.join(args.result_dir, f"V13_EVT_V2_Log_{int(time.time())}.txt")
    logger = Logger(log_file)

    if os.path.exists(args.pcap_dir):
        known_ds, unknown_ds, num_knowns, idx_to_cls = load_dataset_v13_openset(
            args.pcap_dir,
            logger,
            num_unknown_classes=args.num_unknowns
        )

        if len(known_ds) > 0:
            train_v13_evt_v2(known_ds, unknown_ds, num_knowns, idx_to_cls, logger, args.result_dir)
        else:
            logger.log("Error: known dataset is empty after loading.")
    else:
        logger.log(f"Error: Dataset path '{args.pcap_dir}' not found.")

if __name__ == "__main__":
    main()