%% °ë¼à¶½Í¼Ç¶Èë¾ØÕó»ú semi-supervised matrixized graph embedding machine SMGEM
clear;
clc;
load 'AHUT_SGST.mat'
s = 160;z = 7;
data1 = data;

taurange = [0.001 0.01];
lambdarange = [0.01 0.1 ];
knnrange = [2 4];
for ktuning = 1:size(knnrange,2)
    for tautuning = 1:size(taurange,2)
        for lambdatuning= 1:size(lambdarange,2)
            randomsampleindex = randperm(s);
            for i = 1:z
                temp1=data1(:,:,(i-1)*s+1:i*s);
                datatmp(:,:,(i-1)*s+1:i*s)=temp1(:,:,randomsampleindex);
            end
            data = datatmp;
            clear datatmp randomsampleindex
            c = 2;   d = 4; 
            data = data(1:c:end,1:d:end,:);

            tau = taurange(tautuning);
            lambda = lambdarange(lambdatuning);
            knn = knnrange(ktuning);
            labelednum = 10;
            unlabelednum = 60;
            testnum = 90;
            tic;
            X_test=[];
            y_test=[];
            for i=1:z
                temp1=data(:,:,(i-1)*s+(s-testnum)+1:i*s);
                temp2=i*ones(testnum,1);
                X_test=cat(3,X_test,temp1);
                y_test=[y_test;temp2];
            end
            sz_test=size(X_test);
            X_test = reshape(X_test,[sz_test(1)*sz_test(2),sz_test(3)])';
            clear i temp1 temp2 sz_test

            subclassifernum = 0;
            for posnumber=1:(z-1)
                for negnumber=(posnumber+1):z
                    subclassifernum=subclassifernum+1;
                    Xl = cat(3,data(:,:,(posnumber-1)*s+1:(posnumber-1)*s+labelednum), ...
                        data(:,:,(negnumber-1)*s+1:(negnumber-1)*s+labelednum));
                    Xu = cat(3,data(:,:,(posnumber-1)*s+1+labelednum:(posnumber-1)*s+labelednum+unlabelednum), ...
                        data(:,:,(negnumber-1)*s+1+labelednum:(negnumber-1)*s+labelednum+unlabelednum));
                    yl = [ones(labelednum,1);-1*ones(labelednum,1)];
                    X = cat(3,Xl,Xu);
                    
                    [~,L] = semi_adjacency(knn,X);
                    [W,b, obj_value]= ADMM_solver (Xl, Xu, yl, L, tau, lambda);

                    y_hat_test = sign(X_test * W + b);
                    indx = (y_hat_test == 1);
                    y_hat_test(indx) = posnumber;
                    y_hat_test(~indx) = negnumber;
                    
                    predict(:,subclassifernum)=y_hat_test;
                    obj_final(:,subclassifernum)= obj_value;
                    clear Xl Xu X yl L W b y_hat_test indx
                end
            end
            y_pre=[];
            for i=1:testnum*z
                table=tabulate(predict(i,:));
                [~,indx]=max(table(:,2));
                y_pre(i,1)=table(indx);
            end
            fprintf('time = %f\n',toc);
            clear table i indx predict posnumber negnumber subclassifernum X_test
            acc_test=length(find(y_pre == y_test))/length(y_test);
            tuning_result(tautuning,lambdatuning,ktuning)=acc_test;
        end
    end
end
plot(obj_final)
