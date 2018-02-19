SELECT
	fietsersbond_til_join.*, 
	fietsersbondbriders2014_intens0.intens0, 
	fietsersbondbriders2014_intens1.intens1, 
	fietsersbondbriders2014_intens2.intens2, 
	fietsersbondbriders2014_intens3.intens3, 
	fietsersbondbriders2014_intens4.intens4, 
	fietsersbondbriders2014_intens5.intens5, 
	fietsersbondbriders2014_intens6.intens6

INTO fietsersbond_til_join_perdag
FROM fietsersbond_til_join
    left join fietsersbondbriders2014_intens0 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens0.linknummer
	left join fietsersbondbriders2014_intens1 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens1.linknummer
 	left join fietsersbondbriders2014_intens2 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens2.linknummer
 	left join fietsersbondbriders2014_intens3 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens3.linknummer
 	left join fietsersbondbriders2014_intens4 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens4.linknummer
 	left join fietsersbondbriders2014_intens5 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens5.linknummer
 	left join fietsersbondbriders2014_intens6 on  fietsersbond_til_join.linknummer = fietsersbondbriders2014_intens6.linknummer