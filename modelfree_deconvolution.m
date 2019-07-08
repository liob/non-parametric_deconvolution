function [ Fu, Vu, T ] = modelfree_deconvolution(data, mask, varargin)
%This is a matlab version of the generic pixel-by-pixel deconvolution 
%algorithm for perfusion quantification of DCE MRI similarly implemented 
%in [1] Zöllner FG, Weisser G, Reich M, et al. UMMPerfusion: An open source
%software tool towards quantitative MRI perfusion analysis in clinical routine.
%J Digit Imaging 2013;26:344?352.[2] Sourbron S, Biffar A, Ingrisch M, 
%Fierens Y, Luypaert R: PMI: platform for research in medical imaging. 
%Magn Reson Mater Phy 22:539, 2009

%input:
%data: dims - [t x y z]
%   t: temporal dimension; x, y, z: spatial dimensions
%mask: region of interest (ROI) placed over feeding artery to 
%   evaluate arterial input function (AIF) - dims - [x y z]


%optional:
%modelfree_deconvolution(....,'baseline',t1): baseline signal 
%   - default: 1 -> 1st time point
%modelfree_deconvolution(....,'cutoff',t2): cropping of late time 
%   points: default -> no cropping
%modelfree_deconvolution(....,'normalize',rel_sig): normalize 
%   signal to baseline signal - default: 0 = no (else 1 = yes)
%modelfree_deconvolution(....,'hematocrit',Hct): hematocrit level - 
%   default: 0.45
%modelfree_deconvolution(....,'TSVDmax',Slim): threshold for singular 
%   values (S) used in calculation of 
%   pseudo-inverse relative to maximum singular value Smax: S/Smax>Slim
%   default: Slim=0.15
%modelfree_deconvolution(....'dt',dt): temporal resolution of data set in seconds - default: 1s
%modelfree_deconvolution(....,'prefilter',prefilter): smoothing of data - default: 0 -> no smoothing (1 -> smoothing)


%output:
%Fu: Blood Flow in ml/100ml/min
%Vu: Blood Volume in ml/100ml
%T: Mean Transit Time in s

%optional: default parameters
t1=1;
t2=size(data,1);
rel_sig=0;
Hct=.45;
Slim=.15;
dt=1;
prefilter=0;

%numerical parameter
epsilon=1e-12;

if nargin>2
    i=1;
    while(i < numel(varargin))
        switch lower(varargin{i})
            case 'baseline'
                t1=varargin{i+1};
            case 'cutoff'
                t2=varargin{i+1};
            case 'normalize'
                rel_sig=varargin{i+1};
            case 'hematocrit'
                Hct=varargin{i+1};
            case 'tsvdmax'
                Slim=varargin{i+1};
            case 'dt'
                dt=varargin{i+1};
            case 'prefilter'
                prefilter=varargin{i+1};
        end
        i=i+1;
    end
end




%subtraction of baseline signal
sbase=mean(data(1:t1,:,:,:),1);
tmp=data((t1+1):t2,:,:,:)-repmat(mean(data(1:t1,:,:,:),1),[t2-t1 1 1 1]);
data=tmp;
clear 'tmp';
if rel_sig
    data=data./(repmat(sbase,[t2-t1 1 1 1])+1e-6);
end
clear 'sbase';

%aif: mean signal in arterial ROI
aif=mean(data(:,mask==1),2);
aif=aif/(1-Hct);
n=numel(aif);

%optional smoothing
if prefilter==1
    aif0=aif;
    aif2(1)=aif(1)+4*aif(1)+aif(2);

    for i=1:n-2
        aif2(i+1)=aif(i-1+1)+4*aif(i+1)+aif(i+2);
    end

    aif2(n-1+1)=aif(n-2+1)+4*aif(n-1+1)+aif(n-1+1);

    aif2=aif2/6;
    aif=aif2;
    
     data0=data;
    data(1,:,:,:)=data0(1,:,:,:)+4*data0(1,:,:,:)+data0(2,:,:,:);
    
    for i=2:n-1
        data(i,:,:,:)=data0(i-1,:,:,:)+4*data0(i,:,:,:)+data0(i+1,:,:,:);
    end
    
    data(end,:,:,:)=data0(end-1,:,:,:)+4*data0(end,:,:,:)+data0(end,:,:,:);
    data=data/6;
    
    
end


%kernel of AIF using Volterra interpolation
A=zeros(n,n);

for i=1:n-1
    A(i+1,1)=(2*aif(i+1)+aif(i+1-1))/6;
    A(i+1,i+1)=(2*aif(0+1)+aif(1+1))/6;
end

for i=2:n-1
    for j=1:i-1
        A(i+1,j+1)=(2*aif(i-j+1)+aif(i-j))/6+(2*aif(i-j+1)+aif(i-j+1+1))/6;
    end
end

%SVD of kernel of AIF
[U S V]=svd(A);
%truncated SVD: thresholdSchwelle bei 15% des größten Singulärwerts
q=max(find(diag(S)/max(diag(S))>=Slim));
Si=diag(S);

%Inverse of kernel of AIF
Ainv=V(:,1:q)*diag(1./Si(1:q))*U(:,1:q)';



%Impulse Response
I=1/dt*Ainv*reshape(data,n,numel(data)/n);
%Blood Flow
F=reshape(max(I),size(data,2),size(data,3),size(data,4));
%Blood Flow in ml/100ml/min
Fu=F*60*100;
%Blood Volume
V=dt*reshape(sum(I),size(data,2),size(data,3),size(data,4));
%Blood Volume in ml/100ml
Vu=V*100;
%Mean Transit Time in s
T=V./(F+epsilon);

end