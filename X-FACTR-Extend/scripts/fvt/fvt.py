import re
import torch

from fvt import VocabularyTransfer


class FastVocabularyTransfer(VocabularyTransfer):

    def __init__(self):
        super().__init__()

    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        """
        This method establish a mapping between each token of
        the in-domain tokenizer (in_tokenizer) to one or more tokens from
        the general-purpose (gen_tokenizer) tokenizer.

        :param in_tokenizer: Any huggingface tokenizer
        :param gen_tokenizer: Any huggingface tokenizer
        :param kwargs: no kwargs

        :return: A dictionary, having size of the in_tokenizer vocabulary.
         Each key is the index corresponding to a token in the in-tokenizer.
         Values are lists of indexes to the tokens of gen_tokenizer.
        """

        gen_vocab = gen_tokenizer.vocab
        in_vocab = in_tokenizer.vocab
        ngram_vocab = in_tokenizer.ngram_vocab if hasattr(in_tokenizer, 'ngram_vocab') else {}

        tokens_map = {}
        number = 0
        for new_token, new_index in in_vocab.items():
            if new_token in gen_vocab:
                # if the same token exists in the old vocabulary, take its embedding
                old_index = gen_vocab[new_token]
                tokens_map[new_index] = [old_index]
            else:
                number = number + 1
                # if not, tokenize the new token using the old vocabulary
                new_token = re.sub('^(##|Ġ|▁)', '', new_token)
                #new_token = re.sub('^(##|▁)', '', new_token)
                if new_token in ngram_vocab:
                    token_partition = gen_tokenizer.tokenize(new_token.split('‗'), is_split_into_words=True)
                else:
                    token_partition = gen_tokenizer.tokenize(new_token)
                tokens_map[new_index] = [gen_vocab[old_token] for old_token in token_partition]
                ## handle the case where the gen_tokenizer skips the word
                if len(tokens_map[new_index]) == 0:
                    tokens_map[new_index] = [gen_vocab["[UNK]"]]
        print(number, "tokens are not shared")
        return tokens_map

    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        """
        Given a mapping between two tokenizers and a general-purpose model
        trained on gen_tokenizer, this method produces a new embedding matrix
        assigning embeddings from the gen_model.

        :param tokens_map: A mapping between new and old tokens. See tokens_mapping(...)
        :param gen_model: A huggingface model, e.g. bert
        :param kwargs: no kwargs

        :return: (2-d torch.Tensor) An embedding matrix with same size of tokens_map.
        """

        gen_matrix = gen_model.get_input_embeddings().weight
        in_matrix = torch.zeros(len(tokens_map), gen_matrix.shape[1])

        for new_index, old_indices in tokens_map.items():
            #print(gen_matrix[old_indices].shape)
            ## safty check if anything goes wrong
            if len(old_indices) == 0:
                exit()
            old_embedding = torch.mean(gen_matrix[old_indices],0)
            in_matrix[new_index] = old_embedding
        
        # try to randomize the value
        # for new_index, old_indices in tokens_map.items():
        #     embeddings_to_average = []
        #     print(new_index, old_indices)
        #     for idx in old_indices:
        #         ## if it's unknown
        #         if idx == 100:
        #             print("randomize the unk tokens because", new_index, "is not in the vocab")
        #             # Generate a random embedding and normalize it
        #             random_embedding = torch.randn_like(gen_matrix[0])  # Assuming gen_matrix[0] shape for embedding size
        #             normalized_embedding = random_embedding / random_embedding.norm(p=2)  # Normalize with L2 norm
        #             embeddings_to_average.append(normalized_embedding)
        #         else:
        #             embeddings_to_average.append(gen_matrix[idx])  # Use the existing embedding

        #     # Compute the mean of the embeddings (normalized and original)
        #     combined_embedding = torch.mean(torch.stack(embeddings_to_average), dim=0)
        #     in_matrix[new_index] = combined_embedding

        return in_matrix
