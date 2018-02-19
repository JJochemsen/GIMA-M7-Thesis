pgsql2shp -u postgres -P postgres -f C:\Thesis\Data\Output\routes_shape -d thesis "SELECT * from public.fietsersbond_til_join JOIN public.fietsersbondbriders2014_gpsmatch_til USING (linknummer) WHERE fietsersbondbriders2014_gpsmatch_til.routeid = routeid"

SELECT * from public.fietsersbondbriders2014_gpsmatch LEFT JOIN public.fietsersbond_join USING (linknummer);