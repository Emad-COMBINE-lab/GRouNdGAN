import typing

import torch
from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
from torch.utils.data.dataloader import DataLoader


def get_loader(
    genes_no: int,
    file_path: typing.Union[str, typing.List[str]],
    batch_size: int,
    splits: typing.Optional[typing.Dict[str, float]] = None,
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    compression_type: typing.Optional[str] = "gzip",
    multi_read: typing.Optional[bool] = False,
    get_clusters: typing.Optional[bool] = False,
) -> DataLoader:
    """
    Provides an IterableLoader over a Dataset read from given tfrecord files for PyTorch.

    Currently used to create data loaders from the PBMC preprocessed dataset in tfrecord
    from scGAN (Marouf et al.,2020). description parameter and post_process function
    can be modified to accommodate more tfrecord datasets.

    Parameters
    ----------
    genes_no : int
        Number of genes in the expression matrix.
    file_path : typing.Union[str, typing.List[str]]
        Tfrecord file path for reading a single tfrecord (multi_read=False)
        or file pattern for reading multiple tfrecords (ex: /path/{}.tfrecord).
    batch_size : int
        Training batch size.
    splits : typing.Optional[typing.Dict[str, float]], optional
        Dictionary of (key, value) pairs, where the key is used to construct
        the data and index path(s) and the value determines the contribution
        of each split to the batch. Provide when reading from multiple tfrecords
        (multi_read=True), by default None.
    description : typing.Union[typing.List[str], typing.Dict[str, str], None], optional
        List of keys or dict of (key, value) pairs to extract from each record.
        The keys represent the name of the features and the values ("byte", "float", or "int"),
        by default { "indices": None, "values": None, }.
    compression_type : typing.Optional[str], optional
        The type of compression used for the tfrecord. Either 'gzip' or None, by default "gzip".
    multi_read : typing.Optional[bool], optional
        Specifies whether to construct the dataset from multiple tfrecords.
        If True, a file pattern should be passed to file_path, by default False.
    get_clusters : typing.Optional[bool], optional
        If True, the returned data loader will contain the cluster label of cells in
        addition to their gene expression values, by default False.

    Returns
    -------
    DataLoader
        Iterable data loader over the dataset.
    """

    def post_process(
        records: typing.Dict,
    ) -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Transform function to preprocess gene expression. Builds the dense
        gene expression tensor from a sparse representation based on a
        list of gene indices and corresponding expression values.

        Parameters
        ----------
        records : typing.Dict
            Parsed tfrecord.

        Returns
        -------
        typing.Union[typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            A cell's vector of expression levels with or without associated cluster label.
        """
        indices = torch.from_numpy(records["indices"])
        values = torch.from_numpy(records["values"])

        # create dense vector of zeros
        empty = torch.zeros([genes_no])

        # insert expression values in respective indices in the zeroes vector
        indices = indices.reshape([indices.shape[0], 1])
        expression = empty.index_put_(tuple(indices.t()), values)

        # If the number of clusters is not requested, only return expression values
        try:
            cluster = torch.from_numpy(records["cluster_int"])
        except KeyError:
            return expression

        return expression, cluster

    if description is None:
        if get_clusters:
            description = {"indices": None, "values": None, "cluster_int": None}
        else:
            description = {"indices": None, "values": None}

    if multi_read:
        dataset = MultiTFRecordDataset(
            file_path,
            None,
            splits,
            description,
            compression_type=compression_type,
            transform=post_process,
        )
    else:
        dataset = TFRecordDataset(
            file_path,
            None,
            description,
            compression_type=compression_type,
            transform=post_process,
        )

    return DataLoader(dataset, batch_size=batch_size)
