close all;
clear;

DebugShow=false;
L=[4 4];
AcqPoint=[256,256];
N=[AcqPoint(1),AcqPoint(2)*16];
YdireInhomoCoef=0*[0.001,0.001,0.00,0.00]*L(2);% constant, firstorder,second order;
nseg=1;
NumShots=nseg;
spectrum=00;
sw=250000; % unit Hz
OneShotNum=AcqPoint(2)/nseg;
ChirpRvalue=120;
tblip=128e-6;
gammaHz=4.2574e+3 ; % [Hz/G]
matrix_size=[AcqPoint,1];
xScale=linspace(-L(1)/2,L(1)/2,N(1));
yScale=linspace(-L(2)/2,L(2)/2,N(2));
yScale1=linspace(-L(2)/2,L(2)/2,AcqPoint(2));
Ta = (AcqPoint(1)/sw+tblip)*OneShotNum;
ChripTP=Ta/2; % unit second
aSign = -1 ;                                                                                                               
ProcparStruct.np=AcqPoint(1);
ProcparStruct.ne=1;
ProcparStruct.nv=AcqPoint(2);
ProcparStruct.nseg=NumShots;
ProcparStruct.nf=AcqPoint(2);
ProcparStruct.arraydim=1;
ProcparStruct.Rvol=ChirpRvalue;
ProcparStruct.lpe=L(1);
ProcparStruct.lro=L(2);
ProcparStruct.ppe=0;
ProcparStruct.Tp=ChripTP;
ProcparStruct.Gchip=ChirpRvalue/ChripTP/(L(2))/gammaHz;
ProcparStruct.ppe=0;
rfwdth = ProcparStruct.Tp ; % [sec]
GPEe=ProcparStruct.Gchip;
alfa = +aSign * 2*pi * gammaHz * GPEe * rfwdth / (ProcparStruct.lpe) ;
[M r offsets] = spin_system(L,N,spectrum);
imap.a1 = [0 0]; % in Hz/m
imap.a2 = [0. 0.]; % in Hz/(m^2)
imap.ar = 0; % in Hz
offsets = inhomogeneise(offsets,r,N,imap);
M(:,2) = 1;
M(:,3) = 0;

H=load('F:\Matlab项目\Dissertation\Dataset\HCP_brain.mat');
H=cell2mat(struct2cell(H));

slicenum=900;
nrow=AcqPoint(2);
ncol=AcqPoint(1);
GoodImageAll=zeros(slicenum,nrow,ncol);
RoFFTAll=zeros(slicenum,nrow,ncol);
PhaseMapAll=zeros(slicenum,nrow/2,ncol);

Num=1;
for m=1:9  
    for i=41:140    
        Hb=H(i,:,:);     
        Hb=reshape(Hb,[256,256])'; 
        Hb=imresize(Hb,N);    
        M(:,2) = M(:,2) .* Hb(:);
        SpinDensity=Hb;   
        x=linspace(-L(1)/2,L(1)/2,N(1));
        xacq= -AcqPoint(1)/2*1/L(1):1/L(1):(AcqPoint(1)/2-1)*1/L(1);
        [X,Xacq]=meshgrid(x,xacq);
        y=linspace(-L(2)/2,L(2)/2,N(2));
        Finalryxacq=zeros(AcqPoint(2),AcqPoint(1));
        PhasePara=zeros(nseg,7);
        PhaseMapIdeal=zeros(nseg,AcqPoint(1), AcqPoint(2)/nseg);
        EvenOddLinear=zeros(1,1);
        EvenOddConstant=zeros(1,1);
        
        for j=1:1
            EvenOddLinear(j)=0.3*rand(1,1);
            EvenOddConstant(j)=0.5*rand(1,1);
        end
        
        GoodImage=zeros([AcqPoint(2),AcqPoint(1)]);
        for k=1:nseg
            p=0*rand(1,7);
            MotionImag=SpinDensity;%.*MotionPhaseMap        
            SegRandom=0.00*L(2)/100*rand(1,1);
            Tempyacq=(-L(2)/2+(k-1)*L(2)/(AcqPoint(2))+SegRandom): L(2)/(AcqPoint(2)/nseg):((-L(2)/2+(k-1)*L(2)/(AcqPoint(2))+SegRandom)+(AcqPoint(2)/nseg-1)*(L(2)/(AcqPoint(2)/nseg)));        
            B0y=YdireInhomoCoef(1)*y.^0+YdireInhomoCoef(2)*y.^1+YdireInhomoCoef(3)*y.^2+YdireInhomoCoef(4)*y.^3;
            B0acq=YdireInhomoCoef(1)*Tempyacq.^0+YdireInhomoCoef(2)*Tempyacq.^1+YdireInhomoCoef(3)*Tempyacq.^2+YdireInhomoCoef(4)*Tempyacq.^3;        
            [Y, TempYacq]=ndgrid(y,Tempyacq);
            [B0Y, B0Yacq]=ndgrid(B0y,B0acq);       
            Temprxyacq = (MotionImag*exp(1i*alfa*(((Y+B0Y)-(TempYacq+B0Yacq)).^2-(TempYacq+B0Yacq).^2)));
            GoodImage(k:nseg:end,:)=permute(Temprxyacq,[2,1]);
            MotionPhaseMap=polyval2(p, y, x);
            MotionPhaseMap=imresize(MotionPhaseMap,size(Temprxyacq));
            MotionPhaseMapComplex=1*exp(sqrt(-1)*MotionPhaseMap);
            PhaseMapIdeal(k,:,:)=MotionPhaseMap;
            Temprxyacq=Temprxyacq.*MotionPhaseMapComplex;
            Temprxyacq=permute(Temprxyacq,[2,1]);
            Temprxyacq=FFTXSpace2KSpace(Temprxyacq,2);      
            Img=FFTKSpace2XSpace(Temprxyacq,2);
            Imodd=Img(1:2:end,:);
            Imeven=Img(2:2:end,:);
            if(mod(nseg,2)==1)
                if(mod(k,2)==1)
                    %方式1
                    EvenOddLinear(1)
                    EvenOddConstant(1)
                    Imeven=Imeven.*(ones(AcqPoint(2)/2/nseg,1)*exp(1i*2*pi*(EvenOddLinear(1)*linspace(-L(1)/2,L(1)/2,AcqPoint(1))+EvenOddConstant(1))));                    
                    map=angle(ones(AcqPoint(2)/2/nseg,1)*exp(1i*2*pi*(EvenOddLinear(1)*linspace(-L(1)/2,L(1)/2,AcqPoint(1))+EvenOddConstant(1))));
%                     %方式2
%                     map = ones(AcqPoint(2)/2/nseg,1)*(2*pi*EvenOddLinear(1)*linspace(-L(1)/2,L(1)/2,AcqPoint(1))+EvenOddConstant(1));                
%                     emap = cos(map)+1i*sin(map);
%                     Imeven = Imeven.*emap;
                    
                    figure(777);imagesc(abs(map));
                    PhaseMapAll(Num,:,:)=map;
                    
                    Img(2:2:end,:)=Imeven;
                else
                    Imodd=Imodd.*(ones(AcqPoint(2)/2/nseg,1)*exp(1i*(EvenOddLinear(1)*linspace(-L(1)/2,L(1)/2,AcqPoint(1))+EvenOddConstant(1))));
                    Img(1:2:end,:)=Imodd;
                end
            else
                Imeven=Imeven.*(ones(AcqPoint(2)/2/nseg,1)*exp(1i*2*pi*(EvenOddLinear(1)*linspace(-L(1)/2,L(1)/2,AcqPoint(1))+EvenOddConstant(1))));
                Img(2:2:end,:)=Imeven;
            end        
            Temprxyacq=FFTXSpace2KSpace(Img,2);      
            Temprxyacq=FFTKSpace2XSpace(Temprxyacq,2);
            Temprxyacq=permute(Temprxyacq,[2,1]);
            MotionPhaseMap1=polyval2(p, y, x);
            MotionPhaseMap1=imresize(MotionPhaseMap1,size(Temprxyacq));
            MotionPhaseMapComplex1=1*exp(-0*sqrt(-1)*MotionPhaseMap1);
            Temprxyacq=Temprxyacq.*MotionPhaseMapComplex1;
            Temprxyacq=permute(Temprxyacq,[2,1]);
            Temprxyacq=FFTXSpace2KSpace(Temprxyacq,2);
            Temprxyacq=Temprxyacq/max(abs(Temprxyacq(:)));       
            noiseLevel=0.0*0.015/sqrt(128);
            Finalryxacq(k:nseg:end,:)=Temprxyacq+noiseLevel*(randn(size(Temprxyacq)));
        end
        GchirpUnitGauss=ProcparStruct.Gchip;
        FovSPen=ProcparStruct.lpe;
        ChripTP=ProcparStruct.Tp;
        rfwdth = ChripTP ; % [sec]
        [InvAWhole,tmpAFinal]=calcInvA(alfa,L(2),size(Finalryxacq,1),0,-aSign,0,0.9);%ShiftPE is set to 0,as Amir has done it?
        FinalryxacqROFFT=FFTKSpace2XSpace(Finalryxacq,2);
        FinalryxacqROFFT=FinalryxacqROFFT/max(abs(FinalryxacqROFFT(:)))*100;
        GoodImage=GoodImage/max(abs(GoodImage(:)))*100;
        ky1RelativePos=0;
        [tmpInvAZHalfOdd,tmpAFinalOdd]=calcInvA(alfa,L(2),size(Finalryxacq,1)/2,0,-aSign,ky1RelativePos,0.9);
        [tmpInvAZHalfEven,tmpAFinalEven]=calcInvA(alfa,L(2),size(Finalryxacq,1)/2,0,-aSign,ky1RelativePos+0.5,0.9);

        GoodImageAll(Num,:,:)=GoodImage;
        RoFFTAll(Num,:,:)=FinalryxacqROFFT;
        InvAWholeAll=InvAWhole;
        tmpAFinalAll=tmpAFinal;
        tmpInvAZHalfOddAll=tmpInvAZHalfOdd;
        tmpInvAZHalfEvenAll=tmpInvAZHalfEven;


        disp([num2str(Num) ' is ok']);
        Num=Num+1;    
    end   
end

AWhole = InvAWholeAll;  
AFinal = tmpAFinalAll; 
AEven = tmpInvAZHalfEvenAll; 
AOdd = tmpInvAZHalfOddAll; 
Good = GoodImageAll; 
Dis = RoFFTAll; 
Map = PhaseMapAll;
save('LinearMap_0.3_0.5.mat','Good','Dis','AWhole','AFinal','AEven','AOdd','Map');
