import numpy as np
import pandas as pd
from collections import defaultdict
import sys

class Evaluator:
    """
    Evaluates the logs of the experiment and computes accuracy, NLL, and now RMSE 
    by comparing the empirical participant distribution vs the model's predicted distribution 
    per instance.
    """

    def __init__(self, experiment_log, num_parameters=6, min_data_points=5):
        """
        Initialize the evaluator.

        Args:
            experiment_log (list): List of dictionaries containing log entries.
            num_parameters (int): Number of free parameters in the model (for BIC).
            min_data_points (int): Minimum number of data points (participant responses) 
                                   required to include an instance in RMSE computations.
        """
        self.experiment_log = experiment_log
        self.num_parameters = num_parameters
        self.min_data_points = min_data_points  # NEW CODE FOR RMSE

    def compute_metrics(self, session_num=None):
        """
        Compute accuracy, NLL, BIC, and the new RMSE for each explanation condition.
        Returns:
            dict: A dictionary of aggregated metrics separated by explanation condition.
        """
        # This will hold accuracy & NLL metrics
        metrics = defaultdict(list)

        # We also gather data for RMSE at the instance level
        # instance_data[condition][instance_id] = {
        #     "participant_counts": defaultdict(int),  # frequency of each class by participants
        #     "model_prob_list": [],                   # list of predicted probability dicts
        #     "n_participants": 0                      # how many times this instance has appeared
        # }
        instance_data = defaultdict(lambda: defaultdict(lambda: {
            "participant_counts": defaultdict(int),
            "model_prob_list": [],
            "n_participants": 0
        }))

        all_classes = set()
        XAIType = "none"
        i = 0  # Trial counter

        # Predefine trial counts for session slicing, if needed
        # (Adjust logic to match your code/conditions.)
        train_trials_per_session_w_expl = 10  # e.g., 10 train w/ Explanation
        train_trials_per_session_no_expl = 10  # e.g., 10 train w/o Explanation
        test_trials_per_session = 18 * 2  # e.g., 9 test w & w/o Explanation => 18 total
        total_trials_per_session = train_trials_per_session_w_expl + train_trials_per_session_no_expl + test_trials_per_session

        for log in self.experiment_log:
            if log["Step"].lower() != "infer":
                continue  # Skip feedback steps

            # If your code identifies XAI type differently, adapt as needed
            if log["explanation"] is not None:
                XAIType = "importance"

            i += 1
            if session_num:
                # Suppose session_num is 1-based
                start_threshold = (session_num - 1) * total_trials_per_session
                end_threshold = session_num * total_trials_per_session
                if i <= start_threshold:
                    continue
                if i > end_threshold:
                    break

            # We only look at "Test" trials (or adapt if you want Train too)
            if log["trialType"] != "Test":
                continue

            tested_w_xai = log.get("tested_w_xai", "Unknown")
            participant_response = log.get("participant_response")
            ai_prediction = log.get("ai_prediction")
            model_response = log.get("response")
            instance_id = log.get("instance_index")

            if participant_response is None or ai_prediction is None or model_response is None:
                continue

            # Identify all possible classes from model_response if it's a dict
            if isinstance(model_response, dict):
                all_classes.update(model_response.keys())

            # --- Compute Standard Metrics (Accuracy, NLL) ---
            if isinstance(model_response, dict):
                labels = list(model_response.keys())
                raw = np.array([model_response[l] for l in labels], dtype=float)

                # --- Normalize safely ---
                # clip negatives (optional, but robust)
                raw = np.clip(raw, 0.0, None)
                s = raw.sum()
                if s <= 0:
                    probs = np.ones_like(raw) / len(raw)
                else:
                    probs = raw / s

                prob_map = dict(zip(labels, probs))

                # --- Sampling for accuracy (kept) ---
                model_response_label = np.random.choice(labels, p=probs)

                # --- NLL (no sampling, correct) ---
                participant_prob = prob_map.get(participant_response, 0.0)
                nll = -np.log(max(participant_prob, 1e-10))

            else:
                # Deterministic model
                model_response_label = model_response
                participant_prob = 1.0 if model_response_label == participant_response else 0.0
                nll = -np.log(max(participant_prob, 1e-10))


            model_participant_match = int(model_response_label == participant_response)
            model_ai_match = int(model_response_label == ai_prediction)
            participant_ai_match = int(participant_response == ai_prediction)
            nll = -np.log(participant_prob + 1e-10)

            # Store standard metrics
            metrics[tested_w_xai].append({
                "model_participant_match": model_participant_match,
                "model_ai_match": model_ai_match,
                "participant_ai_match": participant_ai_match,
                "nll": nll
            })

            # --- Collect data for RMSE (per instance) ---
            # 1) Update participant_counts for the chosen participant_response
            instance_data[tested_w_xai][instance_id]["participant_counts"][participant_response] += 1
            instance_data[tested_w_xai][instance_id]["n_participants"] += 1

            # 2) Collect the model's predicted probability distribution
            if isinstance(model_response, dict):
                instance_data[tested_w_xai][instance_id]["model_prob_list"].append(model_response)
            else:
                # If your model outputs a single label, treat it like a 1-hot distribution
                # or skip it. For example:
                label_dist = defaultdict(float)
                label_dist[model_response_label] = 1.0
                instance_data[tested_w_xai][instance_id]["model_prob_list"].append(label_dist)

        # Summarize accuracy + NLL and compute BIC
        aggregated = self._aggregate_metrics(metrics)



        # Now compute RMSE across instance IDs for each condition
        # We’ll store them in aggregated[condition]["rmse_model_participant"]
        all_classes = sorted(all_classes)  # fix an ordering for classes
        for condition, inst_dict in instance_data.items():
            # print(inst_dict)
            # sys.exit()
            rmse = self._compute_rmse_for_condition(inst_dict, all_classes)
            aggregated[condition]["rmse_model_participant"] = rmse


        return aggregated

    def _aggregate_metrics(self, metrics):
        """
        Aggregate the collected metrics (accuracy, NLL, BIC).
        """
        aggregated = {}
        for tested_w_xai, entries in metrics.items():
            n = len(entries)
            total_nll = sum(e["nll"] for e in entries)
            if n > 0:
                bic = self.num_parameters * np.log(n) + 2 * total_nll
            else:
                bic = float('nan')

            aggregated[tested_w_xai] = {
                "accuracy_model_participant": np.mean([e["model_participant_match"] for e in entries]) if n>0 else float('nan'),
                "accuracy_model_ai": np.mean([e["model_ai_match"] for e in entries]) if n>0 else float('nan'),
                "accuracy_participant_ai": np.mean([e["participant_ai_match"] for e in entries]) if n>0 else float('nan'),
                "nll_model_participant": np.mean([e["nll"] for e in entries]) if n>0 else float('nan'),
                "bic": bic
            }
        return aggregated

    # NEW CODE FOR RMSE
    def _compute_rmse_for_condition(self, inst_dict, all_classes):
        """
        Given a dictionary of instance-level data (inst_dict) for a single condition,
        compute the RMSE between the empirical participant distribution and the 
        average model probability distribution, across all instance_ids that have 
        at least self.min_data_points responses.
        """
        per_instance_errors = []

        for instance_id, data in inst_dict.items():
            count_dict = data["participant_counts"]
            model_prob_list = data["model_prob_list"]
            n = data["n_participants"]

            if n < self.min_data_points:
                # Skip if not enough data
                continue

            # 1) Compute empirical participant distribution
            #    i.e. frequency of each class / total
            participant_dist = {}
            for c in all_classes:
                participant_dist[c] = count_dict.get(c, 0) / float(n)


            # 2) Compute average model distribution
            #    average each probability across model_prob_list
            avg_model_dist = {}
            num_prob_vectors = len(model_prob_list)
            if num_prob_vectors == 0:
                # No model data for this instance
                continue

            for c in all_classes:
                # sum up probability c across all prob dicts
                total_prob_c = sum(prob_dict.get(c, 0.0) for prob_dict in model_prob_list)
                avg_model_dist[c] = total_prob_c / float(num_prob_vectors)

            # 3) Compute squared differences
            sum_sq_diff = 0.0
            for c in all_classes:
                diff = participant_dist[c] - avg_model_dist[c]
                sum_sq_diff += diff * diff

            # 4) We'll store the MSE for this instance, then sqrt later
            per_instance_errors.append(sum_sq_diff / len(all_classes))

        if len(per_instance_errors) == 0:
            return float('nan')

        # Finally the overall RMSE for the condition
        mse = np.mean(per_instance_errors)
        rmse = np.sqrt(mse)

        return rmse

    def print_metrics(self, aggregated_metrics):
        """
        Print the aggregated metrics in a readable format.
        """
        for tested_w_xai, metrics in aggregated_metrics.items():
            print(f"Tested w/ XAI: {tested_w_xai}")
            print(f"  Accuracy (Model vs Participant): {metrics.get('accuracy_model_participant', float('nan')):.2f}")
            print(f"  Accuracy (Model vs AI): {metrics.get('accuracy_model_ai', float('nan')):.2f}")
            print(f"  Accuracy (Participant vs AI): {metrics.get('accuracy_participant_ai', float('nan')):.2f}")
            print(f"  NLL (Model vs Participant): {metrics.get('nll_model_participant', float('nan')):.2f}")
            print(f"  BIC: {metrics.get('bic', float('nan')):.2f}")
            if "rmse_model_participant" in metrics:
                print(f"  RMSE (Participant Dist vs Model Dist): {metrics['rmse_model_participant']:.4f}")
            print()
