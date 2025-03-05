"""Main script to call functions from other modules."""

import eda
import modeling


def main():
    """Main function to orchestra data loading, processing, and visualization.

    Args:
        None.

    Returns:
        int: Exit code.

    """
    try:
        print("Loading data...")
        data = eda.load_data()
        print("Data loaded successfully.\n")

        print("Calculating average budget imbalance...")
        avg_budget = eda.return_average_budget(data)
        print(f"Average budget imbalance: {avg_budget:.4f} GtCO2\n")

        print("Visualizing budget imbalance over time...")
        eda.visualize_budget(data)

        print("Creating features and training models...")
        models, scores = modeling.train_all(data, exclude=["budget imbalance"])
        print("Models trained successfully.\n")

        print("Model scores:")
        for component, score in scores.items():
            print(f"{component}: {score:.4f}")

        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except IOError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
