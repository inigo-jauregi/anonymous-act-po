import re
from typing import Dict, Type, Any, Optional

from .ctrl_translation import CtrlTranslationModel, CtrlTranslationModelDpo, CtrlTranslationModelCpo
from .ctrl_translation_pivot import CtrlTranslationPivotModel
from .ctrl_bloom import CtrlBloomModel
from .ctrl_qwen import CtrlQwenModel, CtrlQwenModelDpo, CtrlQwenModelCpo
from .ctrl_aws_bedrock import CtrlAwsBedrockModel


class ModelFactory:
    """Factory class for creating appropriate model instances based on pretrained model names."""

    # Model mapping based on patterns in model names
    MODEL_PATTERNS = {
        # NLLB patterns
        r'.*nllb.*': {'cross_entropy': CtrlTranslationModel, 'dpo': CtrlTranslationModelDpo,
                      'cpo': CtrlTranslationModelCpo},
        r'.*facebook/nllb.*': {'cross_entropy': CtrlTranslationModel, 'dpo': CtrlTranslationModelDpo,
                               'cpo': CtrlTranslationModelCpo},

        # BLOOM patterns
        r'.*bloom.*': {'cross_entropy': CtrlBloomModel},
        r'.*bigscience.*bloom.*': {'cross_entropy': CtrlBloomModel},
        r'.*bigscience/bloom.*': {'cross_entropy': CtrlBloomModel},

        # Qwen patterns
        r'.*qwen.*': {'cross_entropy': CtrlQwenModel, 'dpo': CtrlQwenModelDpo, 'cpo': CtrlQwenModelCpo},
        r'.*Qwen.*': {'cross_entropy': CtrlQwenModel, 'dpo': CtrlQwenModelDpo, 'cpo': CtrlQwenModelCpo},
        r'.*Qwen-.*': {'cross_entropy': CtrlQwenModel, 'dpo': CtrlQwenModelDpo, 'cpo': CtrlQwenModelCpo},

        # AWS Bedrock patterns (Claude models)
        r'.*anthropic\.claude.*': {'cross_entropy': CtrlAwsBedrockModel},
        r'.*bedrock.*': {'cross_entropy': CtrlAwsBedrockModel},
        r'.*mistral\.mistral-large.*': {'cross_entropy': CtrlAwsBedrockModel},
    }

    @classmethod
    def identify_model_class(cls, model_name: str, objective: bool) -> Type:
        """
        Identify the appropriate model class based on the model name.

        Args:
            model_name: The pretrained model name/path

        Returns:
            The appropriate model class

        Raises:
            ValueError: If no matching model class is found
        """
        model_name_lower = model_name.lower()

        for pattern, model_class in cls.MODEL_PATTERNS.items():
            if re.search(pattern, model_name_lower, re.IGNORECASE):
                return model_class[objective]

        # If no pattern matches, raise an error with helpful information
        available_patterns = list(cls.MODEL_PATTERNS.keys())
        raise ValueError(
            f"Could not identify model class for '{model_name}'. "
            f"Available patterns: {available_patterns}"
        )

    @classmethod
    def create_model(cls,
                     **kwargs) -> Any:
        """
        Create a model instance based on the model name.

        Args:
            model_name: The pretrained model name/path
            **kwargs: Additional parameters to override defaults

        Returns:
            An instance of the appropriate model class
        """
        model_class = cls.identify_model_class(kwargs['model_name'], kwargs['objective'])

        # If the model class is the standard CtrlTranslationModel, but we flag it as a pivot model, use CtrlTranslationPivotModel
        if model_class == CtrlTranslationModel and kwargs['pivot_lng'] is not None:
            model_class = CtrlTranslationPivotModel

        return model_class(**kwargs)

    @classmethod
    def get_supported_models(cls) -> Dict[str, Type]:
        """
        Get a dictionary of supported model patterns and their corresponding classes.

        Returns:
            Dictionary mapping patterns to model classes
        """
        return cls.MODEL_PATTERNS.copy()

    @classmethod
    def add_model_pattern(cls, pattern: str, model_class: Type, default_params: Optional[Dict] = None):
        """
        Add a new model pattern to the factory.

        Args:
            pattern: Regex pattern to match model names
            model_class: The model class to instantiate
            default_params: Default parameters for this model class
        """
        cls.MODEL_PATTERNS[pattern] = model_class
        if default_params:
            cls.DEFAULT_PARAMS[model_class] = default_params


# Example usage functions
def create_model_from_name(**kwargs):
    """Convenience function to create a model using the factory."""
    return ModelFactory.create_model(**kwargs)


def identify_model_type(model_name: str) -> str:
    """Convenience function to identify model type as string."""
    model_class = ModelFactory.identify_model_class(model_name)
    return model_class.__name__


# # Usage examples:
# if __name__ == "__main__":
#     # Example model names
#     test_models = [
#         'facebook/nllb-200-distilled-600M',
#         './pretrained_lms/bigscience-bloom-7b1',
#         './pretrained_lms/Qwen-Qwen3-32B',
#         'anthropic.claude-3-haiku-20240307-v1:0'
#     ]
#
#     for model_name in test_models:
#         try:
#             model_class = ModelFactory.identify_model_class(model_name)
#             print(f"Model: {model_name} -> Class: {model_class.__name__}")
#
#             # Create an instance (you would provide real src_lng and tgt_lng)
#             # model = ModelFactory.create_model(model_name, 'en', 'fr')
#
#         except ValueError as e:
#             print(f"Error for {model_name}: {e}")
