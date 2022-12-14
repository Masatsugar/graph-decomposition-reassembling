import os
import pickle
from datetime import datetime
from typing import List

from gensim.models import word2vec

from . import features
from .helper import write_to_sdf

BASE_DIR = os.path.abspath(__file__)


class Mol2Vec:
    def __init__(
        self,
        model_path=None,
        radius=2,
        unseen="UNK",
        method="skip-gram",
        window=10,
        vector_size=100,
        threshold=3,
        epochs=1000,
        corpus_name=None,
        save_model=True,
        n_jobs=1,
    ):
        self.radius = radius
        self.model_path = model_path
        self.n_jobs = n_jobs
        self.unseen = unseen
        self.sentences_list = None
        self.vec_list = None
        self.window = window
        self.vector_size = vector_size
        self.threshold = threshold
        self.method = method
        self.epochs = epochs
        self.corpus_name = corpus_name
        self._repr = f"{self.method}_radius{self.radius}_vec{self.vector_size}_window{self.window}_min.pkl"
        if corpus_name is None:
            now = datetime.now()
            self.corpus_name = now.strftime("%Y%m%d_%H%M%S_corpus")

        if self.model_path is not None:
            self.model = word2vec.Word2Vec.load(model_path)
        else:
            self.model = None

        if save_model:
            self.output_file = os.path.join("corpus", self.corpus_name, self._repr)
        else:
            self.output_file = None

    def __repr__(self):
        return f"<{self.__class__.__name__}({self._repr})>"

    def fit(self, mols):
        if self.model is None:
            self.generate_corpus_from_mols(mols, "corpus")
            # input_file = os.path.join(BASE_DIR, f"corpus/{self.corpus_name}")
            input_file = f"corpus/{self.corpus_name}/corpus"
            if self.threshold:
                input_file = input_file + f"_threshold{self.threshold}"

            self.train(
                input_file=input_file,
                method="skip-gram",
                vector_size=self.vector_size,
                window=self.window,
                min_count=1,
                epochs=self.epochs,
            )

    def fit_transform(self, mols):
        self.fit(mols)
        return self.transform(mols)

    def transform(self, mols):
        if self.model is None:
            raise ValueError(
                "This instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        self.sentences_list = [
            features.mol2alt_sentence(mol, self.radius) for mol in mols
        ]
        self.vec_list = features.sentences2vec(
            self.sentences_list, self.model, unseen=self.unseen
        )
        return self.vec_list

    def mols_to_sdf(self, mols, filename):
        # Need to check prefix
        write_to_sdf(mols, filename)
        print(f"Saved sdf in {filename}.")

    def generate_corpus_from_mols(self, mols, output_file):
        """save sdf file from mols and convert it into corpus file.

        Parameters
        ----------
        mols
        output_file

        Returns
        -------

        """
        output_file = output_file.split(".")[0]
        # path = os.path.join(__file__, "corpus")
        path = f"corpus/{self.corpus_name}"
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, output_file)
        write_to_sdf(mols, file_path + ".sdf")
        self.generate_corpus(input_file=file_path + ".sdf", output_file=file_path)

    def generate_corpus(self, input_file, output_file=None, out_corpus=None):
        """Generate corpus file

        Parameters
        ----------
        input_file:
            File format: .sdf, .mol, .smi and so on.
        output_file:
            output file path
        out_corpus:

        Returns
        -------

        """
        features.generate_corpus(
            input_file, output_file, r=self.radius, n_jobs=self.n_jobs
        )
        if self.threshold:
            if out_corpus is None:
                out_corpus = output_file + "_threshold" + str(self.threshold)
            features.insert_unk(
                corpus=output_file,
                out_corpus=out_corpus,
                threshold=self.threshold,
                uncommon=self.unseen,
            )
        else:
            self.unseen = None

    def train(
        self,
        input_file,
        method="skip-gram",
        vector_size=100,
        window=10,
        min_count=1,
        epochs=100,
    ):
        """

        Parameters
        ----------
        input_file:
            Corpus file
        method
        vector_size
        window
        min_count
        epochs

        Returns
        -------

        """
        self.model = features.train_word2vec_model(
            infile_name=input_file,
            outfile_name=self.output_file,
            method=method,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            n_jobs=self.n_jobs,
            epochs=epochs,
        )
        return self

    def save(self, model_path):
        if self.model is not None:
            print("save to", model_path)
            with open(model_path, "wb") as f:
                pickle.dump(self, f)

    def load(self, model_path):
        with model_path.open("rb") as f:
            obj = pickle.load(f)
        return obj

    def load_model(self, model_path):
        self.model = word2vec.Word2Vec.load(model_path)
        self.model_path = model_path
        print(f"load {model_path} successfully.")

    def save_model(self, model_path):
        if self.model is not None:
            print("save to", model_path)
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
