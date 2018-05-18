SELECT linknummer, count(*) AS int0 
INTO briders2014_int0
FROM fietsersbondbriders2014_gpsmatch 
WHERE fietsersbondbriders2014_gpsmatch.weekdag = 0
GROUP BY linknummer
        
