'''
Base class for loss functions in CtrlPost.
'''


class BaseLoss:
    """
    Base class for loss functions in CtrlPost.
    """

    def __init__(self, **kwargs):
        """
        Initialize the loss function with the given configuration.

        Args:
            config (dict): Configuration dictionary for the loss function.
        """
        self.kwargs = kwargs

    def compute_loss(self, predictions, targets):
        """
        Compute the loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.

        Returns:
            torch.Tensor: Computed loss value.
        """
        raise NotImplementedError("Subclasses must implement this method.")