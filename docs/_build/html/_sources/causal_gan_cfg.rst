.. code-block:: ini

    [EXPERIMENT]
    output directory = results/GRouNdGAN
    device = cuda ; we will let the program choose what is available
    checkpoint  ; set value to use a trained model

        [Preprocessing]
        10x = True
        raw = data/raw/PBMC/
        validation set size = 1000 
        test set size = 1000
        annotations = data/raw/PBMC/barcodes_annotations.tsv
        min cells = 3 ; genes expressed in less than 3 cells are discarded
        min genes = 10 ; cells with less than 10 genes expressed are discarded
        library size = 20000 ; library size used for library-size normalization
        louvain res = 0.15 ; Louvain clustering resolution (higher resolution means finding more and smaller clusters)
        highly variable number = 1000 ; number of highly variable genes to identify

        [GRN Preparation]
        TFs = data/raw/Homo_sapiens_TF.csv
        k = 15 ; k is the number of top most important TFs per gene to include in the GRN 
        Inferred GRN = data/processed/PBMC/inferred_grnboost2.csv

        ; "top" for selecting top k edges
        ; "pos ctr" for generating positive control GRNs (even indices 0, 2, 4... = top 1, 3, 5, ...)
        ; "neg ctr" for generating negative control GRNs (odd indices 1, 3, 5... = top 2, 4, 6, ...)
        ; note that k has to be a pair number for strategy=ctr
        strategy = top 

        [Data]
        train = data/processed/PBMC/PBMC68k_train.h5ad
        validation = data/processed/PBMC/PBMC68k_validation.h5ad
        test = data/processed/PBMC/PBMC68k_test.h5ad
        number of genes = 1000
        
        ; this causal graph is a pickled nested dictionary
        ; nested dictionary keys are gene indices
        ; the dictionary is of this form:
        ; {381: {51, 65, 353, 664, 699},
        ; 16: {21, 65, 353, 605, 699},
        ; ...
        ; 565: {18, 51, 65, 552, 650}}
        ; In this example, 381, 16, and 565 are gene indices in the input dataset
        ; Each key's (gene's) value is the indiced of its regulating TFs in the input dataset
        ; A tutorial will be made available in the future.
        
        causal graph = data/processed/PBMC/causal_graph.pkl

        [Generation]
        number of cells to generate = 10000
        generation path ; will save to [Experiment]/output directory/simulated.h5ad if left undefined

        [Evaluation]
        simulated data path ; will use [Generation]/generation path if left undefined
        plot tsne = True # Note: has to be true in order to run miLISI
        compute euclidean distance = True
        compute cosine distance = True 
        compute rf auroc = True
        compute MMD = True
        compute miLISI = True # plot tsne has to be True for this to work
        
        [GRN Benchmarking]
        grn to benchmark = path/to/inferred/grn.csv
        ground truth save path = data/generated/
        plots save path = notebooks/generated/
        compute precision at k = False
        k = 5
        compute pr = True
        
        [Perturbation]
        save dir = data/generated/

        ; tfs to perturb and perturbation values are paired. The two lists need to be of the same size.
        tfs to perturb ;= TF names separated by a space (ex: IRF8 CEBPA)
        perturbation values ;= Float values separated by a space to set respecive TF in tfs to perturb (ex: 0 100.2) 

        [Model]
        type = causal GAN
        noise per gene = 1
        depth per gene = 3
        width per gene = 2
        critic layers = 1024 512 256
        labeler layers = 2000 2000 2000
        latent dim = 128 ; noise vector dimensions
        lambda = 10 ; regularization hyper-parameter for gradient penalty


        [Training]
        batch size = 1024 
        critic iterations = 5 ; iterations to train the critic for each iteration of the generator
        maximum steps = 1000000
        labeler and antilabeler training intervals = 1

            [Optimizer]
            ; coefficients used for computing running averages of gradient and its square 
            beta1 = 0.5
            beta2 = 0.9

            [Learning Rate]
            generator initial = 0.001
            generator final = 0.0001
            critic initial = 0.001
            critic final = 0.001
            labeler = 0.0001
            antilabeler = 0.0001


            [Logging]
            summary frequency = 10000
            plot frequency = 10000
            save frequency = 100000

        [CC Model]
        type = GAN ; Non-conditional single-cell RNA-seq GAN
        generator layers = 256 512 1024
        critic layers = 1024 512 256
        latent dim = 128 ; noise vector dimensions
        lambda = 10 ; regularization hyper-parameter for gradient penalty


        [CC Training]
        batch size = 128 
        critic iterations = 5 ; iterations to train the critic for each iteration of the generator
        maximum steps = 200000

            [CC Optimizer]
            ; coefficients used for computing running averages of gradient and its square 
            beta1 = 0.5
            beta2 = 0.9

            [CC Learning Rate]
            generator initial = 0.0001
            generator final = 0.00001
            critic initial = 0.0001
            critic final = 0.00001

            [CC Logging]
            summary frequency = 10000
            plot frequency = 10000
            save frequency = 100000