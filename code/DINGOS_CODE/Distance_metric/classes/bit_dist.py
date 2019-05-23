import numpy as np
import time

class bit_comparison_count:
    def __init__(self,query, library_matrix):
        self.query = query*2
        self.library_matrix = library_matrix + self.query
        self.hamming_loss_vector = np.zeros(self.library_matrix.shape[0])

    def one_in_both(self):
        both = np.where(self.library_matrix == 3)
        library_index,sorted_unique_value, counts = np.unique(both[0],return_index=True,return_counts=True)
        self.hamming_loss_vector[library_index] += counts

    def one_in_query(self):
        query_one = np.where(self.library_matrix ==2)
        library_index,sorted_unique_value, counts = np.unique(query_one[0],return_index=True, return_counts=True)
        self.hamming_loss_vector[library_index] += counts

    def one_in_library(self):
        library_one = np.where(self.library_matrix==1)
        library_index,sorted_unique_value, counts = np.unique(library_one[0],return_index=True, return_counts=True)
        self.hamming_loss_vector[library_index] += counts

    def zero_in_both(self):
        zero_both = np.where(self.library_matrix == 0)
        library_index, sorted_unique_value, counts = np.unique(zero_both[0],return_index=True, return_counts=True)
        self.hamming_loss_vector[library_index] += counts

    def hamming_loss(self):
        self.one_in_query()
        self.one_in_library()
        del self.library_matrix
        self.hamming_loss_vector = self.hamming_loss_vector/self.query.shape