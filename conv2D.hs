import Control.Parallel.Strategies (using, parListChunk, parList, rseq, rdeepseq, rpar, dot)
import Control.Parallel (pseq)
import System.Environment (getArgs)
import System.CPUTime (getCPUTime)

-- функция для получения размеров изображения
bounds :: [[a]] -> (Int, Int)
bounds img = (length img, length (head img))

-- свертка относительно точки (y, x) изображения
convolve :: Num a => [[a]] -> [[a]] -> Int -> Int -> a
convolve img kernel y x = sum [(img !! (y + dy) !! (x + dx)) * (kernel !! dy !! dx) | dy <- [0 .. fst (bounds kernel) - 1], dx <- [0 .. snd (bounds kernel) - 1]]

-- свертка по всему изображению (поддерживаются прямоугольные ядра)
-- convolveImage :: Num a => [[a]] -> [[a]] -> [[a]]
convolveImage img kernel = 
  let conv = convolve img kernel 
      m = fst (bounds img)
      n = snd (bounds img)
      k_m = fst (bounds kernel)
      k_n = snd (bounds kernel)
  in [[conv i j | j <- [0 .. (n - k_n)]] | i <- [0 .. (m - k_m)]] `using` parListChunk 64 rdeepseq

generateImg :: Int -> Int -> a -> [[a]]
generateImg m n val = replicate m (replicate n val)

main :: IO ()
main = do
  -- 4 аргумента: ширина, высота, значение, размер ядра (m n val k)
  args <- getArgs
  let 
    t = map (read::String->Int) args
    -- не придмал как поумнее сделать (может pattern-matching нужен какой-нибудь)
    img = generateImg (t !! 0) (t !! 1) (t !! 2)
    kernel = generateImg (t !! 3) (t !! 3) (t !! 2)
    result = convolveImage img kernel

  -- pseq чтобы иниициализировать изображение до засечки
  start <- img `pseq` getCPUTime
  -- pseq чтобы заставить result вычислиться (нужно т.к. lazy evaluation)
  end <- result `pseq` getCPUTime
  let time = end - start
  print (fromInteger time / 1e12)
  
