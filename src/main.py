from f1_aggregate_results import aggregate_and_save_scores
from f2_create_chart import create_heatmap_chart


def main():
    aggregate_and_save_scores()
    create_heatmap_chart()

if __name__ == "__main__":
    main()