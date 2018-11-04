double precision function mean(a, n)
  double precision, intent(in) :: a(n)
  integer i
  mean = 0
  do i=1, n
    mean = mean + a(i)
  enddo
  mean = mean/n
end function mean

double precision function stddev(a, n)
  double precision, intent(in) :: a(n)
  double precision mu, mean
  integer i
  mu = mean(a, n)
  stddev = 0
  do i=1, n
    stddev = stddev + (a(i)-mu)**2
  enddo
  stddev = sqrt(stddev/(n-1))
end function stddev

double precision function corr(a, n)
  double precision, intent(in) :: a(n)
  double precision mu, sig, mean, stddev, ct
  integer i, k
  mu = mean(a, n)
  sig = stddev(a, n)
  corr = 0
  do k=1,n
    ct = 0
    do i=1, n-k
      ct = ct + (a(i)-mu)*(a(i+k)-mu)
    enddo
    ct = ct/sig**2/(n-k)
    if (ct .le. 0) then
      exit
    endif
    corr = corr + 2*ct
  enddo
  corr = corr + 1
end function corr
