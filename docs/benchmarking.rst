Benchmarking
============

.. figure:: _static/workflow.svg
   :alt: Benchmarking workflow using GRouNdGAN
   :width: 600
   :align: center
   :class: with-border

   GRN inference benchmarking workflow using GRouNdGAN.

Simulated datasets
------------------


You need 2 elements to benchmark GRN inference methods: 

.. rst-class:: numbered

1. Gene Expression Data [rows (genes) x columns (cells)]

2. Ground Truth GRN

   - Contains two columns:

     1. TF (regulators)
     2. Gene (target)

   - Rows represent imposed edges directed from column 1: TF (regulators) to column 2: Gene (target).


Here, we provide simulated scRNA-seq datasets (100k cells x 1000 genes (including TFs)) and their corresponding ground truth GRNs. The ground truth GRN is the GRN imposed onto GRouNdGAN to generate the simulated dataset. Each gene in the imposed GRN is regulated by 15 TFs (identified using GRNBoost2 (Moerman et al., 2018) on experimental data). 

.. note:: 

    Although this (potentially non-causal) GRN is imposed by GRouNdGAN, it is imposed in a causal manner and represents the causal data generating graph of the simulated data.

.. note:: 
    Feel free to reduce dataset size if you intend to use fewer cells for benchmarking.

BoneMarrow (Paul et al., 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reference dataset: Differentiation of hematopoietic stem cells to different lineages from mouse bone marrow (GEO accession number: GSE72857). 

- `BoneMarrow Expression Data (H5AD) <https://nextcloud.computecanada.ca/index.php/s/aekSWdwociMfQ9p/download>`_

- `BoneMarrow Expression Data (CSV) <https://nextcloud.computecanada.ca/index.php/s/y3bMbarAxNQHJ2b/download>`_

- `BoneMarrow Ground Truth GRN (CSV) <https://nextcloud.computecanada.ca/index.php/s/sNiSmwYi9QBR3Rq/download>`_


PBMC-ALL (Zheng et al., 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reference dataset: Human peripheral blood mononuclear cell (PBMC Donor A) dataset from `10x Genomics <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a>`_.

- `PBMC-ALL Expression Data (H5AD) <https://nextcloud.computecanada.ca/index.php/s/m6eKj8i7qyydp2P/download>`_

- `PBMC-ALL Expression Data (CSV) <https://nextcloud.computecanada.ca/index.php/s/WJTQDXMKMHx2ZTN/download>`_

- `PBMC-ALL Ground Truth GRN (CSV) <https://nextcloud.computecanada.ca/index.php/s/WrQ8r3EZgtG4FX5/download>`_


PBMC-CTL
~~~~~~~~
Reference dataset: Dataset corresponding to the most common cell type (CD8+ Cytotoxic T-cells) in PBMC-All (Zheng et al., 2017).

- `PBMC-CTL Expression Data (H5AD) <https://nextcloud.computecanada.ca/index.php/s/pgQxmJ5LNrMtNBD/download>`_

- `PBMC-CTL Expression Data (CSV) <https://nextcloud.computecanada.ca/index.php/s/aoHG3EqTznjPxw9/download>`_

- `PBMC-CTL Ground Truth GRN (CSV) <https://nextcloud.computecanada.ca/index.php/s/8c7PdnQpSq8oXPL/download>`_

Dahlin (Dahlin et al., 2018)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reference dataset: Hematopoietic dataset corresponding to the scRNA-seq (10x Genomics) profiles of mouse bone marrow hematopoietic stem and progenitor cells (HSPCs) differentiating towards different lineages from  GEO (accession number: GSE107727).

- `Dahlin Expression Data (H5AD) <https://nextcloud.computecanada.ca/index.php/s/WXLKasKsjrGaEAR/download>`_

- `Dahlin Expression Data (CSV) <https://nextcloud.computecanada.ca/index.php/s/wWcQzbQKnMPM6ZS/download>`_

- `Dahlin Ground Truth GRN (CSV) <https://nextcloud.computecanada.ca/index.php/s/zgz2wZTg8q4EoXx/download>`_


Tumor-ALL (Han et al., 2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reference dataset: Malignant cells as well as cells in the tumor microenvironment (called Tumor-All dataset here) from 20 fresh core needle biopsies of follicular lymphoma patients from `cellxgene <https://cellxgene.cziscience.com/collections/968834a0-1895-40df-8720-666029b3bbac>`_.

- `Tumor-ALL Expression Data (H5AD) <https://nextcloud.computecanada.ca/index.php/s/AAZ6rLAA4iyBQnT/download>`_

- `Tumor-ALL Expression Data (CSV) <https://nextcloud.computecanada.ca/index.php/s/E6SDHqkMPrDBtr4/download>`_

- `Tumor-ALL Ground Truth GRN (CSV) <https://nextcloud.computecanada.ca/index.php/s/L7PRLBMWwcMz2rs/download>`_

Tumor-malignant
~~~~~~~~~~~~~~~
Reference dataset: Dataset corresponding to cells labelled as "malignant" in the Tumor-ALL (Han et al., 2022) dataset. 

- `Tumor-malignant Expression Data (H5AD) <https://nextcloud.computecanada.ca/index.php/s/xmKioe8ddFRing4/download>`_

- `Tumor-malignant Expression Data (CSV) <https://nextcloud.computecanada.ca/index.php/s/SswbZmnZwrSkWW5/download>`_

- `Tumor-malignant Ground Truth GRN (CSV) <https://nextcloud.computecanada.ca/index.php/s/q2RejFdPoCkXtZs/download>`_

------------

.. admonition:: Help us expand this list: request GRouNdGAN training on new GRNs and reference datasets

    We are eager to grow this list and welcome your contributions. If you would like us to train GRouNdGAN on a new reference dataset with different GRNs, please submit a request by opening an `issue <https://github.com/Emad-COMBINE-lab/GRouNdGAN/issues/new>`__ on our GitHub repository. Be sure to include a link to the reference dataset in your request.

    If you have trained GRouNdGAN on a new dataset and would like to contribute to our collection, we encourage you to open a pull request.






References
----------
Paul, F., Arkin, Y., Giladi, A., Jaitin, D. A., Kenigsberg, E., Keren-Shaul, H., Winter, D. R., Lara-Astiaso, D., Gury, M., Weiner, A., David, E., Cohen, N., Lauridsen, F. K. B., Haas, S., Schlitzer, A., Mildner, A., Ginhoux, F., Jung, S., Trumpp, A., . . . Tanay, A. (2015). Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell, 163(7), 1663–1677. https://doi.org/10.1016/j.cell.2015.11.013

Zheng, G., Terry, J. M., Belgrader, P., Ryvkin, P., Bent, Z., Wilson, R. J., Ziraldo, S. B., Wheeler, T. D., McDermott, G. P., Zhu, J., Gregory, M., Shuga, J., Montesclaros, L., Underwood, J. G., Masquelier, D. A., Nishimura, S. Y., Schnall-Levin, M., Wyatt, P., Hindson, C. M., . . . Bielas, J. H. (2017). Massively parallel digital transcriptional profiling of single cells. Nature Communications, 8(1). https://doi.org/10.1038/ncomms14049

Moerman, T., Aibar, S., González-Blas, C. B., Simm, J., Moreau, Y., Aerts, J., & Aerts, S. (2018). GRNBoost2 and Arboreto: efficient and scalable inference of gene regulatory networks. Bioinformatics, 35(12), 2159–2161. https://doi.org/10.1093/bioinformatics/bty916

Dahlin, J. S., Hamey, F. K., Pijuan-Sala, B., Shepherd, M., Lau, W. W., Nestorowa, S., ... & Wilson, N. K. (2018). A single-cell hematopoietic landscape resolves 8 lineage trajectories and defects in Kit mutant mice. Blood, The Journal of the American Society of Hematology, 131(21), e1-e11.

Han, G., Deng, Q., Marques-Piubelli, M. L., Dai, E., Dang, M., Ma, M. C. J., ... & Green, M. R. (2022). Follicular lymphoma microenvironment characteristics associated with tumor cell mutations and MHC class II expression. Blood cancer discovery, 3(5), 428-443.