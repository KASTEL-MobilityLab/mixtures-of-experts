from typing import Tuple
import torch


def calculate_metrics(target, predictions, confidences, uncertainty_threshold: float) -> Tuple[float, float, float]:
    inaccuracy_map = predictions != target
    certain_map = (1 - confidences) < uncertainty_threshold
    uncertain_map = ~certain_map

    accurate_certain = torch.sum((~inaccuracy_map) & certain_map).item()
    certain = torch.sum(certain_map).item()
    uncertain_inaccurate = torch.sum(inaccuracy_map & uncertain_map).item()
    inaccurate = torch.sum(inaccuracy_map).item()

    p_accurate_given_certain = accurate_certain / certain if certain > 0 else 0
    p_uncertain_given_inaccurate = uncertain_inaccurate / \
        inaccurate if inaccurate > 0 else 0
    pavpu = (accurate_certain + uncertain_inaccurate) / \
        (certain + inaccurate) if (certain + inaccurate) > 0 else 0

    return p_accurate_given_certain, p_uncertain_given_inaccurate, pavpu


def calculate_ece(confidences, predictions, labels, num_bins=15):
    """Calculate Expected Calibration Error (ECE)"""
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = torch.logical_and(confidences > bin_lower,
                                   confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (
                predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].float().mean()
            ece += (torch.abs(avg_confidence_in_bin -
                              accuracy_in_bin) * prop_in_bin).item()

    return ece


def calculate_mce(confidences, predictions, labels, num_bins=15):
    """Calculate Maximum Calibration Error (MCE)"""
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        confidences_greater = confidences > bin_lower
        confidences_lower = confidences <= bin_upper
        in_bin = torch.logical_and(confidences_greater, confidences_lower)
        prop_in_bin = in_bin.int().sum()
        if prop_in_bin > 0:
            predictions_in_bin = predictions[in_bin]
            labels_in_bin = labels[in_bin]
            accuracy_in_bin = (predictions_in_bin ==
                               labels_in_bin).float().mean()
            avg_confidence_in_bin = confidences[in_bin].float().mean()
            mce = max(mce, torch.abs(
                avg_confidence_in_bin - accuracy_in_bin).item())

    return mce


def get_confidences(prediction: torch.Tensor, outputs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    softmax = torch.softmax(outputs, dim=2)
    uncertainty = torch.std(softmax, dim=0)

    max_uncertainty = uncertainty.max()
    if max_uncertainty > 0:
        uncertainty = uncertainty / max_uncertainty

    uncertainty_label = torch.zeros(
        (image.shape[0], image.shape[2], image.shape[3]), device=image.device)

    for class_index in range(outputs.shape[2]):
        mask = prediction == class_index
        uncertainty_for_class = uncertainty[:, class_index, :, :]
        uncertainty_label = torch.where(
            mask, uncertainty_for_class, uncertainty_label)

    confidences = 1 - uncertainty_label
    return confidences


def compute_evu(prediction: torch.Tensor, outputs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    softmax = torch.softmax(outputs, dim=2)
    uncertainty = torch.std(softmax, dim=0)

    max_uncertainty = uncertainty.max()
    if max_uncertainty > 0:
        uncertainty = uncertainty / max_uncertainty

    uncertainty_label = torch.zeros(
        (image.shape[0], image.shape[2], image.shape[3]), device=image.device)

    for class_index in range(outputs.shape[2]):
        mask = prediction == class_index
        uncertainty_for_class = uncertainty[:, class_index, :, :]
        uncertainty_label = torch.where(
            mask, uncertainty_for_class, uncertainty_label)

    return uncertainty_label


def get_predictive_entropy(mean_softmax: torch.Tensor) -> torch.Tensor:

    # Sum over classes (dim=1)
    predictive_entropy = - \
        torch.sum(mean_softmax * torch.log(mean_softmax + 1e-10), dim=1)

    return predictive_entropy  # Shape: [batch_size, img_height, img_width]


def compute_mutual_information(predictive_entropy: torch.Tensor, softmax_probs: torch.Tensor) -> torch.Tensor:
    # Shape: [n_forward_passes, batch_size, img_height, img_width]
    individual_entropies = - \
        torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=2)

    # Shape: [batch_size, img_height, img_width]
    mean_entropy = torch.mean(individual_entropies, dim=0)

    # Shape: [batch_size, img_height, img_width]
    mutual_information = predictive_entropy - mean_entropy

    return mutual_information


def clamp_and_log_values(tensor, name="tensor", log_event=True):
    # Count the number of values greater than 1
    num_adjusted = torch.sum(tensor > 1).item()

    if num_adjusted > 0:
        print(f"{name}: {num_adjusted} values greater than 1 were adjusted to 1.")
        # Clamp values greater than 1 to 1
        tensor = torch.clamp(tensor, max=1)

    return tensor


def enable_dropout(model: torch.nn.Module):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
