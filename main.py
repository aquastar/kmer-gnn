from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
from data_processing import *
from models import *
import time
import matplotlib.pyplot as plt
import time
import sys


batch_size = 32
epoch_num = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv('important_10mer_binary.csv', index_col=0)
antibiotics = data['Antibiotic'].unique().tolist()
print(antibiotics)
datasets = split_by_antibiotic(data, antibiotics)

graph_creation_start = time.time()
adj_file =   sys.argv[1] #'adjacency_matrix_hamming_distance.csv'
adj = pd.read_csv(adj_file, index_col=0)
# adj = adj.drop(['Antibiotic', 'MIC'], axis=1)
# adj = adj.drop(['Antibiotic', 'MIC'], axis=0)

node_index = []
for kmer in adj.columns:
    node_index.append(kmer)

# edge_index, edge_attr = graph_creation(adj, node_index)
edge_index, edge_attr = graph_creation(adj, node_index, threshold=.5)
graph_creation_end = time.time()
print('Graph creation time: ', graph_creation_end - graph_creation_start)

# for antibiotic in antibiotics:
for antibiotic in ['COT']:
    print('=============================================================')
    print(f"Processing antibiotic: {antibiotic}")
    dataset = datasets[antibiotic]

    data_processing_start = time.time()

    graphs = []
    for i in range(dataset.shape[0]):
        input = dataset.iloc[i]
        x = []
        for kmer_index in range(dataset.shape[1]-1):
            kmer = node_index[kmer_index]
            x.append([input[kmer]])
        x = torch.tensor(x, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([input['MIC']], dtype=torch.float))
        graphs.append(graph)
    # graphs = DataLoader(graphs, batch_size=1)
    data_processing_end = time.time()
    print('number of graphs: ', len(graphs))

    kfold = KFold(n_splits=5, shuffle=True)
    correct_percentages_gcn = []
    correct_percentages_gatv2 = []
    correct_percentages_appnp = []
    correct_percentages_sage = []
    correct_percentages_mlp = []
    correct_percentages_gat = []
    correct_percentages_lr = []
    correct_percentages_rf = []
    correct_percentages_svr = []
    correct_percentages_xgb = []
    
    gcn_time = []
    sage_time = []
    gatv2_time = []
    mlp_time = []
    gat_time = []
    lr_time = []
    rf_time = []
    svr_time = []
    xgb_time = []
    
    ig = 0
    for train_index, test_index in kfold.split(graphs):
        # print(f'Fold {ig}', end=',')
        ig += 1
        train_set = []
        test_set = []
        for index in train_index:
            train_set.append(graphs[index])
        for index in test_index:
            test_set.append(graphs[index])

        train = DataLoader(train_set, batch_size=batch_size)
        test = DataLoader(test_set, batch_size=batch_size)

        # MLP
        # model_mlp = MLP(1351, 1).to(device)
        # # Define loss criterion and optimizer
        # criterion_mlp = nn.MSELoss()
        # optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=0.0003)

        # mlp_start = time.time()
        # # Training loop
        # for epoch in range(epoch_num):
        #     model_mlp.train()
        #     for instance in train:
        #         instance = instance.to(device)
        #         optimizer_mlp.zero_grad()
        #         out_mlp = model_mlp(instance)
        #         loss_mlp = criterion_mlp(out_mlp.squeeze(), instance.y.squeeze())
        #         loss_mlp.backward()
        #         optimizer_mlp.step()
        # mlp_end = time.time()
        # mlp_time.append(mlp_end - mlp_start)
        # # Evaluation
        # model_mlp.eval()
        # with torch.no_grad():
        #     correct_mlp = 0
        #     for instance in test:
        #         instance = instance.to(device)
        #         out_mlp = model_mlp(instance)
        #         for i in range(out_mlp.size(0)):
        #             if out_mlp[i].item() >= instance.y[i].item() / 2 and out_mlp[i].item() <= instance.y[i].item() * 2:
        #                 correct_mlp += 1
        # correct_percentages_mlp.append(correct_mlp / test_index.size)
        
        # GCN
        # print('GCN', end=',')
        model_gcn = GCNRegression(num_layers=1).to(device)
        criterion_gcn = nn.MSELoss()
        optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.0003)

        gcn_start = time.time()
        for epoch in range(epoch_num):
            model_gcn.train()
            for instance in train:
                instance = instance.to(device)
                optimizer_gcn.zero_grad()
                out_gcn = model_gcn(instance)
                loss_gcn = criterion_gcn(out_gcn.squeeze(), instance.y.squeeze())
                loss_gcn.backward()
                optimizer_gcn.step()
        gcn_end = time.time()
        gcn_time.append(gcn_end - gcn_start)
        model_gcn.eval()
        with torch.no_grad():
            correct_gcn = 0
            for instance in test:
                instance = instance.to(device)
                out_gcn = model_gcn(instance)
                for i in range(out_gcn.size(0)):
                    if out_gcn[i].item() >= instance.y[i].item()/2 and out_gcn[i].item() <= instance.y[i].item()*2:
                        correct_gcn += 1
            correct_percentages_gcn.append(correct_gcn / test_index.size)
            # print(f'GCN percentage: {correct_percentage:.5f}')      
        # break

        # print('SAGE', end=',')
        # model_sage = SAGERegression(num_layers=1).to(device)
        # criterion_sage = nn.MSELoss()
        # optimizer_sage = torch.optim.Adam(model_sage.parameters(), lr=0.0003)
        # sage_start = time.time()

        # for epoch in range(epoch_num):
        #     model_sage.train()
        #     for instance in train:
        #         instance = instance.to(device)
        #         optimizer_sage.zero_grad()
        #         out_sage = model_sage(instance)
        #         loss_sage = F.mse_loss(out_sage.squeeze(), instance.y.squeeze())
        #         loss_sage.backward()
        #         optimizer_sage.step()
        # sage_end = time.time()
        # sage_time.append(sage_end - sage_start)
        # model_sage.eval()
        # with torch.no_grad():
        #     correct_sage = 0
        #     for instance in test:
        #         instance = instance.to(device)
        #         out_sage = model_sage(instance)
        #         for i in range(out_sage.size(0)):
        #             if out_sage[i].item() >= instance.y[i].item()/2 and out_sage[i].item() <= instance.y[i].item()*2:
        #                 correct_sage += 1
        #     correct_percentages_sage.append(correct_sage/test_index.size)


        # print('GAT', end=',')
        # model_gat = GATRegression(num_layers=1).to(device)
        # criterion_gat = nn.MSELoss()
        # optimizer_gat = torch.optim.Adam(model_gat.parameters(), lr=0.0003)
        # gat_start = time.time()
        # for epoch in range(epoch_num):
        #     model_gat.train()
        #     for instance in train:
        #         instance = instance.to(device)
        #         optimizer_gat.zero_grad()
        #         out_gat = model_gat(instance)
        #         loss_gat = F.mse_loss(out_gat.squeeze(), instance.y.squeeze())
        #         loss_gat.backward()
        #         optimizer_gat.step()
        # gat_end = time.time()
        # gat_time.append(gat_end - gat_start)
        # model_gat.eval()
        # with torch.no_grad():
        #     correct_gat = 0
        #     for instance in test:
        #         instance = instance.to(device)
        #         out_gat = model_gat(instance)
        #         for i in range(out_gat.size(0)):
        #             if out_gat[i].item() >= instance.y[i].item()/2 and out_gat[i].item() <= instance.y[i].item()*2:
        #                 correct_gat += 1
        #     correct_percentages_gat.append(correct_gat/test_index.size)
            
            
        # print('GATv2', end=',')
        # model_gatv2 = GATv2Regression(num_layers=1).to(device)
        # criterion_gatv2 = nn.MSELoss()
        # optimizer_gatv2 = torch.optim.Adam(model_gatv2.parameters(), lr=0.0003)
        # gatv2_start = time.time()

        # for epoch in range(epoch_num):
        #     model_gatv2.train()
        #     for instance in train:
        #         instance = instance.to(device)
        #         optimizer_gatv2.zero_grad()
        #         out_gatv2 = model_gatv2(instance)
        #         loss_gatv2 = F.mse_loss(out_gatv2.squeeze(), instance.y.squeeze())
        #         loss_gatv2.backward()
        #         optimizer_gatv2.step()
        # gatv2_end = time.time()
        # gatv2_time.append(gatv2_end - gatv2_start)
        # model_gatv2.eval()
        # with torch.no_grad():
        #     correct_gatv2 = 0
        #     for instance in test:
        #         instance = instance.to(device)
        #         out_gatv2 = model_gatv2(instance)
        #         for i in range(out_gatv2.size(0)):
        #             if out_gatv2[i].item() >= instance.y[i].item()/2 and out_gatv2[i].item() <= instance.y[i].item()*2:
        #                 correct_gatv2 += 1
        #     correct_percentages_gatv2.append(correct_gatv2/test_index.size)

        # print('APPNP')
        # model_appnp = APPNPRegression(num_layers=1).to(device)
        # criterion_appnp = nn.MSELoss()
        # optimizer_appnp = torch.optim.Adam(model_appnp.parameters(), lr=0.0003)
        # appnp_start = time.time()

        # for epoch in range(5):
        #     model_appnp.train()
        #     for instance in train:
        #         instance = instance.to(device)
        #         optimizer_appnp.zero_grad()
        #         out_appnp = model_appnp(instance)
        #         loss_appnp = F.mse_loss(out_appnp.squeeze(), instance.y.squeeze())
        #         loss_appnp.backward()
        #         optimizer_appnp.step()
        # appnp_end = time.time()
        # # appnp_time.append(appnp_end - appnp_start)
        # model_appnp.eval()
        # with torch.no_grad():
        #     correct_appnp = 0
        #     for instance in test:
        #         instance = instance.to(device)
        #         out_appnp = model_appnp(instance)
        #         for i in range(out_appnp.size(0)):
        #             if out_appnp[i].item() >= instance.y[i].item()/2 and out_appnp[i].item() <= instance.y[i].item()*2:
        #                 correct_appnp += 1
        #     correct_percentages_appnp.append(correct_appnp/test_index.size)
        

        # print('BASELINES', end=',')
        
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for instance in train_set:
            X_train.append(instance.x.numpy().flatten())
            Y_train.append(instance.y.item())
        for instance in test_set:
            X_test.append(instance.x.numpy().flatten())
            Y_test.append(instance.y.item())

        lr_start = time.time()
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        lr_end = time.time()
        lr_time.append(lr_end - lr_start)
        # test lr on X_test and Y_test
        correct = 0
        for i in range(len(X_test)):
            pred = lr.predict([X_test[i]])
            if pred >= Y_test[i]/2 and pred <= Y_test[i]*2:
                correct += 1
        correct_percentages_lr.append(correct/len(X_test))
        # print(f'LR percentage: {correct_percentage:.5f}')

        rf_start = time.time()
        rf = RandomForestRegressor()
        rf.fit(X_train, Y_train)
        rf_end = time.time()
        rf_time.append(rf_end - rf_start)
        # test rf on X_test and Y_test
        correct = 0
        for i in range(len(X_test)):
            pred = rf.predict([X_test[i]])
            if pred >= Y_test[i]/2 and pred <= Y_test[i]*2:
                correct += 1
        correct_percentages_rf.append(correct/len(X_test))
        # print(f'RF percentage: {correct_percentage:.5f}')

        svr_start = time.time()
        svr = SVR()
        svr.fit(X_train, Y_train)
        svr_end = time.time()
        svr_time.append(svr_end - svr_start)
        # test svr on X_test and Y_test
        correct = 0
        for i in range(len(X_test)):
            pred = svr.predict([X_test[i]])
            if pred >= Y_test[i]/2 and pred <= Y_test[i]*2:
                correct += 1
        correct_percentages_svr.append(correct/len(X_test))
        # print(f'SVR percentage: {correct_percentage:.5f}')

        xgb_start = time.time()
        xgb = XGBRegressor()
        xgb.fit(X_train, Y_train)
        xgb_end = time.time()
        xgb_time.append(xgb_end - xgb_start)
        # test xgb on X_test and Y_test
        correct = 0
        for i in range(len(X_test)):
            pred = xgb.predict([X_test[i]])
            if pred >= Y_test[i]/2 and pred <= Y_test[i]*2:
                correct += 1
        correct_percentages_xgb.append(correct/len(X_test))
        # print(f'XGB percentage: {correct_percentage:.5f}')
        
        # print('')
        
    print(f'GCN ACC: {np.mean(correct_percentages_gcn):.4f},   STD: {np.std(correct_percentages_gcn):.4f},  time: {np.mean(gcn_time):.4f}')
    # print(f'MLP ACC: {np.mean(correct_percentages_mlp):.4f},   STD: {np.std(correct_percentages_mlp):.4f},  time: {np.mean(mlp_time):.4f}')
    # print(f'GAT ACC: {np.mean(correct_percentages_gat):.4f},   STD: {np.std(correct_percentages_gat):.4f},  time: {np.mean(gat_time):.4f}')
    # print(f'SAGE ACC: {np.mean(correct_percentages_sage):.4f},   STD: {np.std(correct_percentages_sage):.4f},  time: {np.mean(sage_time):.4f}')
    # print(f'GATv2 ACC: {np.mean(correct_percentages_gatv2):.4f},   STD: {np.std(correct_percentages_gatv2):.4f},  time: {np.mean(gatv2_time):.4f}')
    # print(f'APPNP ACC: {np.mean(correct_percentages_appnp):.4f},   STD: {np.std(correct_percentages_appnp):.4f},  time: {np.mean(appnp_time):.4f}')
    print(f'LR ACC: {np.mean(correct_percentages_lr):.4f},  STD: {np.std(correct_percentages_lr):.4f},  time: {np.mean(lr_time):.4f}')
    print(f'RF ACC: {np.mean(correct_percentages_rf):.4f},  STD: {np.std(correct_percentages_rf):.4f},  time: {np.mean(rf_time):.4f}')
    print(f'SVR ACC: {np.mean(correct_percentages_svr):.4f},  STD: {np.std(correct_percentages_svr):.4f},  time: {np.mean(svr_time):.4f}')
    print(f'XGB ACC: {np.mean(correct_percentages_xgb):.4f},  STD: {np.std(correct_percentages_xgb):.4f},  time: {np.mean(xgb_time):.4f}')
