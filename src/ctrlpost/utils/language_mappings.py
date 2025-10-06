
def lang_token_mapping(lang_token, type):

    LANGUAGE_MAP_DICT = {
        'ar': {'NLLB': 'arb_Arab', 'prompt': 'Arabic'},
        'da': {'NLLB': 'dan_Latn', 'prompt': 'Danish'},
        'de': {'NLLB': 'deu_Latn', 'prompt': 'German'},
        'en': {'NLLB': 'eng_Latn', 'prompt': 'English'},
        'es': {'NLLB': 'spa_Latn', 'prompt': 'Spanish'},
        'fr': {'NLLB': 'fra_Latn', 'prompt': 'French'},
        'hi': {'NLLB': 'hin_Deva', 'prompt': 'Hindi'},
        'it': {'NLLB': 'ita_Latn', 'prompt': 'Italian'},
        'ja': {'NLLB': 'jpn_Jpan', 'prompt': 'Japanese'},
        'nl': {'NLLB': 'nld_Latn', 'prompt': 'Dutch'},
        'pl': {'NLLB': 'pol_Latn', 'prompt': 'Polish'},
        'pt': {'NLLB': 'por_Latn', 'prompt': 'Portuguese'},
        'ru': {'NLLB': 'rus_Cyrl', 'prompt': 'Russian'},
    }

    if lang_token in LANGUAGE_MAP_DICT:
        return LANGUAGE_MAP_DICT[lang_token][type]
    else:
        raise ValueError(f"Unsupported language token: {lang_token}")
