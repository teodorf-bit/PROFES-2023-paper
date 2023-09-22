import graphlearning as gl
import pandas as pd
import csv
import numpy as np
import time as t



total_labels = 931 
algorithms = [
              "amle",
              "centered_kernel",
              #"dynamic_label_propagation", # This cannot be used on large datasets
              "modularity_mbo",
              "peikonal",
              "poisson_mbo",
              "randomwalk",
              "sparse_label_propagation",
              "graph_nearest_neighbor",
              "laplace",
              "plaplace",
              "laplace_wnll",
              "laplace_poisson",
              "poisson",
              "volume_mbo"
             ]
ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
datasets = ["saab"]
seeds = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,91,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
for dataset in datasets:
    labels = gl.datasets.load(dataset, labels_only=True)
    W = gl.weightmatrix.knn(dataset, 10, metric='raw')
    D = gl.weightmatrix.knn(dataset, 10, metric='raw', kernel='distance')
    for algorithm in algorithms:
        for seed in seeds:
            for ratio in ratios:
                num_train_per_class = ratio*total_labels
                train_ind = gl.trainsets.generate(labels, rate=round(num_train_per_class))
                train_labels = labels[train_ind]
                class_priors = gl.utils.class_priors(labels)
                if algorithm == "amle":
                    start = t.time()
                    model = gl.ssl.amle(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "centered_kernel":
                    start = t.time()
                    model = gl.ssl.centered_kernel(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "dynamic_label_propagation":
                    start = t.time()
                    model = gl.ssl.dynamic_label_propagation(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "modularity_mbo":
                    start = t.time()
                    model = gl.ssl.modularity_mbo(W)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "peikonal":
                    start = t.time()
                    model = gl.ssl.peikonal(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "poisson_mbo":
                    start = t.time()
                    model = gl.ssl.poisson_mbo(W, class_priors=class_priors, use_cuda=True)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "randomwalk":
                    start = t.time()
                    model = gl.ssl.randomwalk(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "sparse_label_propagation":
                    start = t.time()
                    model = gl.ssl.sparse_label_propagation(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "graph_nearest_neighbor":
                    start = t.time()
                    model = gl.ssl.graph_nearest_neighbor(D, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "laplace":
                    start = t.time()
                    model = gl.ssl.laplace(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "plaplace":
                    start = t.time()
                    model = gl.ssl.plaplace(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "laplace_wnll":
                    start = t.time()
                    model = gl.ssl.laplace(W, reweighting='wnll', class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "laplace_poisson":
                    start = t.time()
                    model = gl.ssl.laplace(W, reweighting='poisson', class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "poisson":
                    start = t.time()
                    model = gl.ssl.poisson(W, solver='gradient_descent', class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                elif algorithm == "volume_mbo":
                    start = t.time()
                    model = gl.ssl.volume_mbo(W, class_priors=class_priors)
                    pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)
                    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
                    time = t.time() - start
                else:
                    print(algorithm)
                    print("No Algorithm with that name")
                    
                header = ["accuracy","dataset","algorithm","seed","ratio","time"]
                data = [accuracy, dataset, algorithm, seed, ratio, time]
                
                np.savetxt("results-"+dataset+"/results-"+str(algorithm)+"-"+str(ratio)+"-"+str(seed)+".csv", data, fmt='%s', delimiter=",")
                

                with open("results-"+dataset+"/results-"+str(algorithm)+"-"+str(ratio)+"-"+str(seed)+".csv", 'w', encoding='utf8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerow(data)
                
                
