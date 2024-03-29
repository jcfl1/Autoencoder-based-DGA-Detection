import pandas as pd
import numpy as np
import re
import math
import json
from nltk import ngrams


# Regex for removing TLDs, including compose TLDs as ".com.fr", ".gov.br", etc
COMPILED_TLD_REGEX = re.compile(r'\.(?:edu|gov|com|co|org|ac|ne|net|mil|int)\.[^\.]+$')

# Preprocessing functions --------------------------------------------------------------------------------------------------------
def remove_tld(domain):
    # Returns input domain string after removing TLD
    match_object = COMPILED_TLD_REGEX.search(domain)
    if match_object is not None:
        return domain[: match_object.start()]

    find = domain.rfind(".")
    if find == -1:
        print(f"[WARNING] A domain without a TLD was found in raw data ({domain})")
        return domain
    return domain[:find]


def get_ngram_frequencies(top_domains, n_list=[3, 4, 5, 6, 7], top_n_domains=100000):
    """
    Returns a dict counting ngram frequencies in "top_domains" pandas Series Object

    Parameters
    ----------
    top_domains : Pandas Series
        Series containing domains sorted by popularity
    n_list: List of ints
        A list containing "n"s from ngrams to be extracted
    top_n_domains: Int
        Number of domains to be used to count ngram frequencies
    """

    top_domains = top_domains[:top_n_domains]
    domains_without_tld_list = top_domains.apply(lambda domain: remove_tld(domain))

    ngram_frequencies = {}
    for n in n_list:
        ngram_frequencies[f"{n}-grams"] = {}

    for domain_without_tld in domains_without_tld_list:
        label_list = domain_without_tld.split(".")
        for label in label_list:
            for n in n_list:
                curr_ngrams = ngrams(label, n)
                for ngram in curr_ngrams:
                    string_ngram = "".join(ngram)
                    if string_ngram in ngram_frequencies[f"{n}-grams"]:
                        ngram_frequencies[f"{n}-grams"][string_ngram] += 1
                    else:
                        ngram_frequencies[f"{n}-grams"][string_ngram] = 1

    return ngram_frequencies


def weight_ngram(ngram_str, dict_ngram_frequencies):
    # Returns the weight of a single ngram to be used to calculate reputation value
    ngram_key = str(len(ngram_str)) + "-grams"
    if ngram_str not in dict_ngram_frequencies[ngram_key]:
        return 0
    return math.log2((dict_ngram_frequencies[ngram_key][ngram_str]) / (len(ngram_str)))


def reputation_value(domain, dict_ngram_frequencies):
    # Retuns reputation value for a entire domain string (after removing TLD)
    domain = remove_tld(domain)
    if domain == "":
        print("[WARNING] A domain without a TLD was found in raw data while trying to calculate reputation value")
        return None

    reputation_value = 0
    n_list = [int(st[0]) for st in dict_ngram_frequencies.keys()]
    label_list = domain.split(".")
    for label in label_list:
        for n in n_list:
            curr_ngrams = ngrams(label, n)
            for ngram in curr_ngrams:
                string_ngram = "".join(ngram)
                if string_ngram in dict_ngram_frequencies[f"{n}-grams"]:
                    reputation_value += weight_ngram(string_ngram, dict_ngram_frequencies)
                else:
                    reputation_value += weight_ngram(string_ngram, dict_ngram_frequencies)
    return reputation_value


def extract_ratios_frequencies_and_lengths(fqdn):
    """
    Returns a 11 element tuple containing: string_char_frequencies, domain_len, max_len_label, max_len_continuous_int,
    max_len_continuous_string, special_freq, special_ratio, int_freq, int_ratio, vowel_freq, vowel_ratio

    Parameters
    ----------
    fqdn : string
        Fully Qualified Domain Name string in lowercase

    """
    domain_len = 0
    string_char_frequencies = {c: 0 for c in "abcdefghijklmnopqrstuvwxyz0123456789"}
    ints, vowels, specials = 0, 0, 0
    max_len_continuous_int, curr_continuous_ints, max_len_continuous_string, curr_continuous_string, max_len_label, curr_label = (
        0,
        0,
        0,
        0,
        0,
        0,
    )

    for c in fqdn:
        domain_len += 1
        curr_label += 1
        # if c.isalpha():
        if "a" <= c <= "z":
            # Increment string_char_frequencies
            string_char_frequencies[c] += 1

            # Adjust lenghts
            curr_continuous_string += 1

            if curr_continuous_string > max_len_continuous_string:
                max_len_continuous_string = curr_continuous_string

            curr_continuous_ints = 0

        elif c.isdigit():
            # Increment string_char_frequencies
            string_char_frequencies[c] += 1
            # Increment ints frequency
            ints += 1

            # Adjust lenghts
            curr_continuous_ints += 1

            if curr_continuous_ints > max_len_continuous_int:
                max_len_continuous_int = curr_continuous_ints

            curr_continuous_string = 0

        else:
            if c == ".":
                # Adjust lenghts (Reset curr_label)
                curr_label = 0
            else:
                # Increment special frequency
                specials += 1

            # Adjust lenghts
            curr_continuous_ints = 0
            curr_continuous_string = 0

        if curr_label > max_len_label:
            max_len_label = curr_label

    for v in "aeiou":
        if v in string_char_frequencies:
            vowels += string_char_frequencies[v]

    return (
        string_char_frequencies,
        domain_len,
        max_len_label,
        max_len_continuous_int,
        max_len_continuous_string,
        specials,
        round(specials / domain_len, 5),
        ints,
        round(ints / domain_len, 5),
        vowels,
        round(vowels / domain_len, 5),
    )


def extract_features(df, domain_col, ngram_frequencies, should_remove_TLD=False):
    """
    Returns the original dataframe with new columns for linguistic features.

    Parameters
    ----------
    df : Dataframe
        Input dataframe
    domain_col : String
        A string representing the name of the column from the input dataframe containing the lowercase query
    top_domains: List like object
        List of domains to extract n-gram reputation value features
    should_remove_TLD : Bool
        Set if you want to obtain linguistic features after removing TLD
    n_list: List of ints
        A list containing "n"s from ngrams to be extracted
    top_n_domains: Int
        Number of domains to be used to count ngram frequencies
    """

    new_df = pd.DataFrame()

    if should_remove_TLD:
        df[f"wo_tld_{domain_col}"] = df.apply(lambda df_aux: remove_tld(df_aux[domain_col]), axis=1)
        domains = df[f"wo_tld_{domain_col}"]
    else:
        domains = df[domain_col]

    # Obtaining domain_len, max_len_label, max_len_continuous_int, max_len_continuous_string, special_freq,
    # special_ratio, int_freq, int_ratio, vowel_freq, vowel_ratio
    (
        string_char_frequencies,
        new_df["domain_len"],
        new_df["max_len_label"],
        new_df["max_len_continuous_int"],
        new_df["max_len_continuous_string"],
        new_df["special_freq"],
        new_df["special_ratio"],
        new_df["int_freq"],
        new_df["int_ratio"],
        new_df["vowel_freq"],
        new_df["vowel_ratio"],
    ) = zip(*domains.map(extract_ratios_frequencies_and_lengths))

    # Obtaining char frequencies for every string char [a-z0-9]
    df_string_char_frequencies = pd.json_normalize(string_char_frequencies)
    df_string_char_frequencies = df_string_char_frequencies.add_suffix("_freq")
    new_df = pd.concat([new_df, df_string_char_frequencies], axis=1)

    # Adding prefix to features obtained without TLD
    if should_remove_TLD:
        new_df = new_df.add_prefix("wo_tld_")

    new_df["reputation_value"] = df.apply(
        lambda df_aux: reputation_value(df_aux[domain_col], ngram_frequencies), axis=1
    )

    df = pd.concat([df, new_df], axis=1)

    return df

def extract_character_level_representation(df, domain_col, max_len=256, should_remove_TLD=True, 
                                           valid_chars='abcdefghijklmnopqrstuvwxyz0123456789.-'):
    
    if should_remove_TLD:
        df[f"wo_tld_{domain_col}"] = df.apply(lambda df_aux: remove_tld(df_aux[domain_col]), axis=1)
        domains = df[f"wo_tld_{domain_col}"]
    else:
        domains = df[domain_col]

    valid_chars_dict = {c:i+1 for i,c in enumerate(valid_chars)}
    
    numeric_domain_representation_list = []
    for domain in domains:
        domain = domain[-max_len:]
        numeric_domain_representation = [valid_chars_dict[c] if c in valid_chars else len(valid_chars)+1 for c in domain]
        # Padding
        num_zeros_to_pad = max_len - len(numeric_domain_representation)
        numeric_domain_representation = np.pad(
            numeric_domain_representation, pad_width=(num_zeros_to_pad, 0), mode='constant', constant_values=0
            )
        numeric_domain_representation_list.append(numeric_domain_representation)
    numeric_domain_representation_array = np.array(numeric_domain_representation_list, dtype=np.int8)
    return numeric_domain_representation_array

