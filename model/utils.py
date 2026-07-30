import matplotlib.pyplot as plt

def plot_activations(activations, syllable_id=None, title=None):
    """
    Plot the 4 muscle activation curves (resp, PCAIA, CT, TA) produced
    by LSTMController over the sequence's timesteps.
 
    Args:
        activations: tensor of shape (1, seq_len, 4) or (seq_len, 4) —
            output from LSTMController.forward(), for a single syllable.
            If batch dimension is present, only the first item is plotted.
        syllable_id: optional int/tensor, the syllable class this came
            from — used in the auto-generated title if `title` isn't given.
        title: optional custom title. Overrides the auto-generated one.
 
    Returns:
        None. Displays the plot.
    """
    # drop batch dim if present, detach from graph, move to numpy
    if activations.dim() == 3:
        activations = activations[0]
    activations = activations.detach().cpu().numpy()
 
    labels = ["resp_activity", "PCAIA_activity", "CT_activity", "TA_activity"]
 
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        plt.plot(activations[:, i], label=label)
 
    plt.xlabel("Timestep (% of syllable duration)")
    plt.ylabel("Activation (0-1)")
    plt.ylim(-0.05, 1.05)
 
    if title:
        plt.title(title)
    elif syllable_id is not None:
        plt.title(f"Muscle activations: syllable class {syllable_id}")
    else:
        plt.title("Muscle activations over time")
 
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_activations_multi(activations_dict):
    """
    Plot muscle activation curves for multiple syllable classes side by
    side, one subplot per class — useful for comparing whether different
    syllable IDs produce visibly different activation patterns.
 
    Args:
        activations_dict: dict mapping syllable_id (int) -> activations
            tensor of shape (1, seq_len, 4) or (seq_len, 4).
 
    Returns:
        None. Displays the plot.
    """
    labels = ["resp_activity", "PCAIA_activity", "CT_activity", "TA_activity"]
    n = len(activations_dict)
 
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
 
    for ax, (syllable_id, activations) in zip(axes, activations_dict.items()):
        if activations.dim() == 3:
            activations = activations[0]
        activations = activations.detach().cpu().numpy()
 
        for i, label in enumerate(labels):
            ax.plot(activations[:, i], label=label)
 
        ax.set_title(f"Syllable {syllable_id}")
        ax.set_xlabel("Timestep")
        ax.set_ylim(-0.05, 1.05)
 
    axes[0].set_ylabel("Activation (0-1)")
    axes[0].legend()
    plt.tight_layout()
    plt.show()
