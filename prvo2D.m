%2 D
clc
clear


hv=20; Fm=10^13; Hr=70; Sa=2; ttm=277; tobs=4;
tk1=(177:100:3600); %čas obsevanja 
tk=repelem(tk1,1,2); % vsak element v tem array zapiše dvakrat ker rabim za v ciklu 
Np=zeros(20,5); % 20x5 elementov v 2 D
Np(:,:,2)=zeros(20,5);% da bi dobili rešitve po času rabimo 3 dimenzijo za čas
F=zeros(70,1); % fluks po x osi, celi sredici 
H=zeros(ttm+49,1); % inicijaliziramo homogenost

%fluks je v x osi cos, max ima na sredini
for i=1:35
  F(i)=Fm*cos(deg2rad(pi*i)/Hr);
  F(i+35)=F(i);
end
F(1:35)=flip(F(1:35));

%ob t=0 je vzorec na sredini reaktorja, Np(t=0,povsod)=0
%gremo dol do dna
    for tt=1:27
        for k=1:5
           for j=1:20
                if tt==1
                Np(j,k,tt)=Sa*F(23+tt+j)*tobs*exp(-Sa*k);
                else
                Np(j,k,tt)=Np(j,k,tt-1)+Sa*F(23+tt+j)*tobs*exp(-Sa*k); 
                end
           end
        end
 H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
    end
    
 Np=rot90(Np,2);  %rotiramo za 180 stopinj 
    %od dno -> do vrha    
for tt=28:77
    for k=1:5
        for j=1:20
            Np(j,k,tt)= Np(j,k,tt-1)+Sa*F(77-tt+j)*tobs*exp(-Sa*k);
        end
    end
 H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
end

%{
    Np=rot90(Np,1); %manjso strano paralelno z gorivni elementi
    %od dno -> vrha
for tt=28:77
    for k=1:5
        for j=1:20
            Np(k,j,tt)= Np(k,j,tt-1)+Sa*F(77-tt+j)*tobs*exp(-Sa*j);
        end
    end
 H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
end
%}
Np=rot90(Np,2); %vecjo strano paralelno z gorivni elementi
%od vrha do dna 
for tt=78:127
    for k=1:5
        for j=1:20
         Np(j,k,tt)= Np(j,k,tt-1)+Sa*F(tt-77+j)*tobs*exp(-Sa*k);
        end
    end
    H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
end

Np=rot90(Np,2);
disp(length(Np(:,1,tt)))
% zdaj za ostali uri obsevanja gremo po tk array 

for ii=1:length(tk)
   % if length(Np(:,1,tt))==20
    %pomeni daljšo strano imamo paralelno z gorivni elementi
        if mod(ii,2)==1
            for tt=(tk(ii)-49):tk(ii)
              for k=1:5
                  for j=1:20
                    Np(j,k,tt)=Np(j,k,tt-1)+Sa*F(tk(ii)-tt+j)*tobs*exp(-Sa*k);
                  end
        H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
              end
            end
        else
        %mod(ii,2)==0
            for tt=(tk(ii)+1):(tk(ii)+50)
                for k=1:5
                    for j=1:20
                       Np(j,k,tt)=Np(j,k,tt-1)+Sa*F(tt-tk(ii)+j)*tobs*exp(-Sa*k);
                    end
        H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
                end 
            end
   
        end
  %{
  else %imamo krajso strano paralelno z gorivni elementi
        for jj=1:length(tk)
            if mod(jj,2)==1
                for tt=(tk(jj)-49):tk(jj)
                    for k1=1:5
                        for j1=1:20
                            Np(k1,j1,tt)=Np(k1,j1,tt-1)+Sa*F(tk(jj)-tt+j)*tobs*exp(-Sa*j1);
                        end
        H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
                    end
                end
            else
        %mod(ii,2)==0
                for tt=(tk(jj)+1):(tk(jj)+50)
                    for k1=1:5
                        for j1=1:20
                            Np(k1,j1,tt)=Np(k1,j1,tt-1)+Sa*F(tt-tk(jj)+j)*tobs*exp(-Sa*j1);
                        end
        H(tt)=(max(max(Np(:,:,tt)),[],'all')-min(min(Np(:,:,tt)),[],'all'))/mean(mean(Np(:,:,tt)));
                    end 
                end
   
              end
        end        
    end
    %}
  Np=rot90(Np,2);
end
 
%plot 

figure(1)
imagesc(Np(:,:,tt))
xlabel('d_v [cm]'); ylabel('h_v [cm]'); colorbar 
title(sprintf('v = %.2f [cm/s]',1/tobs));


figure(2)
plot(H(1:length(H)),'LineWidth',1,'Color','[0.6350, 0.0780, 0.1840]') ; grid on;
xlabel('t [s]'); ylabel('H [/]');
xlim([0 3700])
title(sprintf('v = %.2f [cm/s]',1/tobs));
