function [X_feats,Y]=extractCov3dJ(X)
%%Feature extraction used in Human Action Recognition Using a Temporal Hierarchy
%%of Covariance Descriptors on 3D Joint Locations.In this scenario just one
%%level of C is taken 

X_feats=[];
Y=[];
for i=1:length(X)
   
    for j=1:size(X{i},1)
        mat_tmp=X{i}{j};
        nb_frames=size(mat_tmp,2);
        mat=zeros(size(mat_tmp,1),size(mat_tmp,1));
        for k=1:nb_frames
            S=mat_tmp(:,k);
            mat=mat+(S-mean(S))*(S-mean(S))';
        end
        mat=(1/(nb_frames-1))*mat;
        X_feats=[X_feats;mat(itril(size(mat)))'];
        Y=[Y;i];
    end

perm=randperm(size(X_feats,1));
X_feats=X_feats(perm,:);
Y=Y(perm,:);
end